// This file is part of the SymCC runtime.
//
// The SymCC runtime is free software: you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// The SymCC runtime is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the SymCC runtime. If not, see <https://www.gnu.org/licenses/>.

#include <Runtime.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>
#include <fstream>

#ifndef NDEBUG
#include <chrono>
#endif

#include "Config.h"
#include "GarbageCollection.h"
#include "LibcWrappers.h"
#include "Shadow.h"

#ifndef NDEBUG
// Helper to print pointers properly.
#define P(ptr) reinterpret_cast<void *>(ptr)
#endif

#define FSORT(is_double)                                                       \
  ((is_double) ? Z3_mk_fpa_sort_double(g_context)                              \
               : Z3_mk_fpa_sort_single(g_context))

/* TODO Eventually we'll want to inline as much of this as possible. I'm keeping
   it in C for now because that makes it easier to experiment with new features,
   but I expect that a lot of the functions will stay so simple that we can
   generate the corresponding bitcode directly in the compiler pass. */

namespace {

/// Indicate whether the runtime has been initialized.
std::atomic_flag g_initialized = ATOMIC_FLAG_INIT;

/// The global Z3 context.
Z3_context g_context;

/// The global floating-point rounding mode.
Z3_ast g_rounding_mode;

/// The global Z3 solver.
Z3_solver g_solver; // TODO make thread-local

// Some global constants for efficiency.
Z3_ast g_null_pointer, g_true, g_false;

FILE *g_log = stderr;

#ifndef NDEBUG
[[maybe_unused]] void dump_known_regions() {
  std::cerr << "Known regions:" << std::endl;
  for (const auto &[page, shadow] : g_shadow_pages) {
    std::cerr << "  " << P(page) << " shadowed by " << P(shadow) << std::endl;
  }
}

void handle_z3_error(Z3_context c [[maybe_unused]], Z3_error_code e) {
  assert(c == g_context && "Z3 error in unknown context");
  std::cerr << Z3_get_error_msg(g_context, e) << std::endl;
  assert(!"Z3 error");
}
#endif

Z3_ast build_variable(const char *name, uint8_t bits) {
  Z3_symbol sym = Z3_mk_string_symbol(g_context, name);
  auto *sort = Z3_mk_bv_sort(g_context, bits);
  Z3_inc_ref(g_context, (Z3_ast)sort);
  Z3_ast result = Z3_mk_const(g_context, sym, sort);
  Z3_inc_ref(g_context, result);
  Z3_dec_ref(g_context, (Z3_ast)sort);
  return result;
}

/// The set of all expressions we have ever passed to client code.
std::set<SymExpr> allocatedExpressions;

SymExpr registerExpression(Z3_ast expr) {
  if (allocatedExpressions.count(expr) == 0) {
    // We don't know this expression yet. Record it and increase the reference
    // counter.
    allocatedExpressions.insert(expr);
    Z3_inc_ref(g_context, expr);
  }

  return expr;
}

} // namespace

void _sym_initialize(void) {
  if (g_initialized.test_and_set())
    return;

#ifndef NDEBUG
  std::cerr << "Initializing symbolic runtime" << std::endl;
#endif

  loadConfig();
  initLibcWrappers();
  std::cerr << "This is SymCC running with the simple backend" << std::endl
            << "For anything but debugging SymCC itself, you will want to use "
               "the QSYM backend instead (see README.md for build instructions)"
            << std::endl;

  Z3_config cfg;

  cfg = Z3_mk_config();
  Z3_set_param_value(cfg, "model", "true");
  Z3_set_param_value(cfg, "timeout", "10000"); // milliseconds
  g_context = Z3_mk_context_rc(cfg);
  Z3_del_config(cfg);

#ifndef NDEBUG
  Z3_set_error_handler(g_context, handle_z3_error);
#endif

  g_rounding_mode = Z3_mk_fpa_round_nearest_ties_to_even(g_context);
  Z3_inc_ref(g_context, g_rounding_mode);

  g_solver = Z3_mk_solver(g_context);
  Z3_solver_inc_ref(g_context, g_solver);

  auto *pointerSort = Z3_mk_bv_sort(g_context, 8 * sizeof(void *));
  Z3_inc_ref(g_context, (Z3_ast)pointerSort);
  g_null_pointer = Z3_mk_int(g_context, 0, pointerSort);
  Z3_inc_ref(g_context, g_null_pointer);
  Z3_dec_ref(g_context, (Z3_ast)pointerSort);
  g_true = Z3_mk_true(g_context);
  Z3_inc_ref(g_context, g_true);
  g_false = Z3_mk_false(g_context);
  Z3_inc_ref(g_context, g_false);

  if (g_config.logFile.empty()) {
    g_log = stderr;
  } else {
    g_log = fopen(g_config.logFile.c_str(), "w");
  }
}

Z3_ast _sym_build_integer(uint64_t value, uint8_t bits) {
  auto *sort = Z3_mk_bv_sort(g_context, bits);
  Z3_inc_ref(g_context, (Z3_ast)sort);
  auto *result =
      registerExpression(Z3_mk_unsigned_int64(g_context, value, sort));
  Z3_dec_ref(g_context, (Z3_ast)sort);
  return result;
}

Z3_ast _sym_build_integer128(uint64_t high, uint64_t low) {
  return registerExpression(Z3_mk_concat(
      g_context, _sym_build_integer(high, 64), _sym_build_integer(low, 64)));
}

Z3_ast _sym_build_float(double value, int is_double) {
  auto *sort = FSORT(is_double);
  Z3_inc_ref(g_context, (Z3_ast)sort);
  auto *result =
      registerExpression(Z3_mk_fpa_numeral_double(g_context, value, sort));
  Z3_dec_ref(g_context, (Z3_ast)sort);
  return result;
}

#if DEBUG_CONSISTENCY_CHECK
static std::vector<uint8_t> inputs_;
void pushInputByte(size_t offset, uint8_t value) {
  
  if (inputs_.size() <= offset)
    inputs_.resize(offset + 1);

  inputs_[offset] = value;
}
#endif

Z3_ast _sym_get_input_byte(size_t offset, uint8_t data) {
  static std::vector<SymExpr> stdinBytes;

  if (offset < stdinBytes.size())
    return stdinBytes[offset];

#if DEBUG_CONSISTENCY_CHECK
  pushInputByte(offset, data);
#endif

  auto varName = "stdin" + std::to_string(stdinBytes.size());
  auto *var = build_variable(varName.c_str(), 8);

  stdinBytes.resize(offset);
  stdinBytes.push_back(var);

  return var;
}

Z3_ast _sym_build_null_pointer(void) { return g_null_pointer; }
Z3_ast _sym_build_true(void) { return g_true; }
Z3_ast _sym_build_false(void) { return g_false; }
Z3_ast _sym_build_bool(bool value) { return value ? g_true : g_false; }

Z3_ast _sym_build_neg(Z3_ast expr) {
  return registerExpression(Z3_mk_bvneg(g_context, expr));
}

#define DEF_BINARY_EXPR_BUILDER(name, z3_name)                                 \
  SymExpr _sym_build_##name(SymExpr a, SymExpr b) {                            \
    return registerExpression(Z3_mk_##z3_name(g_context, a, b));               \
  }

DEF_BINARY_EXPR_BUILDER(add, bvadd)
DEF_BINARY_EXPR_BUILDER(sub, bvsub)
DEF_BINARY_EXPR_BUILDER(mul, bvmul)
DEF_BINARY_EXPR_BUILDER(unsigned_div, bvudiv)
DEF_BINARY_EXPR_BUILDER(signed_div, bvsdiv)
DEF_BINARY_EXPR_BUILDER(unsigned_rem, bvurem)
DEF_BINARY_EXPR_BUILDER(signed_rem, bvsrem)
DEF_BINARY_EXPR_BUILDER(shift_left, bvshl)
DEF_BINARY_EXPR_BUILDER(logical_shift_right, bvlshr)
DEF_BINARY_EXPR_BUILDER(arithmetic_shift_right, bvashr)

DEF_BINARY_EXPR_BUILDER(signed_less_than, bvslt)
DEF_BINARY_EXPR_BUILDER(signed_less_equal, bvsle)
DEF_BINARY_EXPR_BUILDER(signed_greater_than, bvsgt)
DEF_BINARY_EXPR_BUILDER(signed_greater_equal, bvsge)
DEF_BINARY_EXPR_BUILDER(unsigned_less_than, bvult)
DEF_BINARY_EXPR_BUILDER(unsigned_less_equal, bvule)
DEF_BINARY_EXPR_BUILDER(unsigned_greater_than, bvugt)
DEF_BINARY_EXPR_BUILDER(unsigned_greater_equal, bvuge)
DEF_BINARY_EXPR_BUILDER(equal, eq)

DEF_BINARY_EXPR_BUILDER(and, bvand)
DEF_BINARY_EXPR_BUILDER(or, bvor)
DEF_BINARY_EXPR_BUILDER(bool_xor, xor)
DEF_BINARY_EXPR_BUILDER(xor, bvxor)

DEF_BINARY_EXPR_BUILDER(float_ordered_greater_than, fpa_gt)
DEF_BINARY_EXPR_BUILDER(float_ordered_greater_equal, fpa_geq)
DEF_BINARY_EXPR_BUILDER(float_ordered_less_than, fpa_lt)
DEF_BINARY_EXPR_BUILDER(float_ordered_less_equal, fpa_leq)
DEF_BINARY_EXPR_BUILDER(float_ordered_equal, fpa_eq)

#undef DEF_BINARY_EXPR_BUILDER

Z3_ast _sym_build_fp_add(Z3_ast a, Z3_ast b) {
  return registerExpression(Z3_mk_fpa_add(g_context, g_rounding_mode, a, b));
}

Z3_ast _sym_build_fp_sub(Z3_ast a, Z3_ast b) {
  return registerExpression(Z3_mk_fpa_sub(g_context, g_rounding_mode, a, b));
}

Z3_ast _sym_build_fp_mul(Z3_ast a, Z3_ast b) {
  return registerExpression(Z3_mk_fpa_mul(g_context, g_rounding_mode, a, b));
}

Z3_ast _sym_build_fp_div(Z3_ast a, Z3_ast b) {
  return registerExpression(Z3_mk_fpa_div(g_context, g_rounding_mode, a, b));
}

Z3_ast _sym_build_fp_rem(Z3_ast a, Z3_ast b) {
  return registerExpression(Z3_mk_fpa_rem(g_context, a, b));
}

Z3_ast _sym_build_fp_abs(Z3_ast a) {
  return registerExpression(Z3_mk_fpa_abs(g_context, a));
}

Z3_ast _sym_build_not(Z3_ast expr) {
  return registerExpression(Z3_mk_bvnot(g_context, expr));
}

Z3_ast _sym_build_not_equal(Z3_ast a, Z3_ast b) {
  return registerExpression(Z3_mk_not(g_context, Z3_mk_eq(g_context, a, b)));
}

Z3_ast _sym_build_bool_and(Z3_ast a, Z3_ast b) {
  Z3_ast operands[] = {a, b};
  return registerExpression(Z3_mk_and(g_context, 2, operands));
}

Z3_ast _sym_build_bool_or(Z3_ast a, Z3_ast b) {
  Z3_ast operands[] = {a, b};
  return registerExpression(Z3_mk_or(g_context, 2, operands));
}

Z3_ast _sym_build_float_ordered_not_equal(Z3_ast a, Z3_ast b) {
  return registerExpression(
      Z3_mk_not(g_context, _sym_build_float_ordered_equal(a, b)));
}

Z3_ast _sym_build_float_ordered(Z3_ast a, Z3_ast b) {
  return registerExpression(
      Z3_mk_not(g_context, _sym_build_float_unordered(a, b)));
}

Z3_ast _sym_build_float_unordered(Z3_ast a, Z3_ast b) {
  Z3_ast checks[2];
  checks[0] = Z3_mk_fpa_is_nan(g_context, a);
  checks[1] = Z3_mk_fpa_is_nan(g_context, b);
  return registerExpression(Z3_mk_or(g_context, 2, checks));
}

Z3_ast _sym_build_float_unordered_greater_than(Z3_ast a, Z3_ast b) {
  Z3_ast checks[3];
  checks[0] = Z3_mk_fpa_is_nan(g_context, a);
  checks[1] = Z3_mk_fpa_is_nan(g_context, b);
  checks[2] = _sym_build_float_ordered_greater_than(a, b);
  return registerExpression(Z3_mk_or(g_context, 2, checks));
}

Z3_ast _sym_build_float_unordered_greater_equal(Z3_ast a, Z3_ast b) {
  Z3_ast checks[3];
  checks[0] = Z3_mk_fpa_is_nan(g_context, a);
  checks[1] = Z3_mk_fpa_is_nan(g_context, b);
  checks[2] = _sym_build_float_ordered_greater_equal(a, b);
  return registerExpression(Z3_mk_or(g_context, 2, checks));
}

Z3_ast _sym_build_float_unordered_less_than(Z3_ast a, Z3_ast b) {
  Z3_ast checks[3];
  checks[0] = Z3_mk_fpa_is_nan(g_context, a);
  checks[1] = Z3_mk_fpa_is_nan(g_context, b);
  checks[2] = _sym_build_float_ordered_less_than(a, b);
  return registerExpression(Z3_mk_or(g_context, 2, checks));
}

Z3_ast _sym_build_float_unordered_less_equal(Z3_ast a, Z3_ast b) {
  Z3_ast checks[3];
  checks[0] = Z3_mk_fpa_is_nan(g_context, a);
  checks[1] = Z3_mk_fpa_is_nan(g_context, b);
  checks[2] = _sym_build_float_ordered_less_equal(a, b);
  return registerExpression(Z3_mk_or(g_context, 2, checks));
}

Z3_ast _sym_build_float_unordered_equal(Z3_ast a, Z3_ast b) {
  Z3_ast checks[3];
  checks[0] = Z3_mk_fpa_is_nan(g_context, a);
  checks[1] = Z3_mk_fpa_is_nan(g_context, b);
  checks[2] = _sym_build_float_ordered_equal(a, b);
  return registerExpression(Z3_mk_or(g_context, 2, checks));
}

Z3_ast _sym_build_float_unordered_not_equal(Z3_ast a, Z3_ast b) {
  Z3_ast checks[3];
  checks[0] = Z3_mk_fpa_is_nan(g_context, a);
  checks[1] = Z3_mk_fpa_is_nan(g_context, b);
  checks[2] = _sym_build_float_ordered_not_equal(a, b);
  return registerExpression(Z3_mk_or(g_context, 2, checks));
}

Z3_ast _sym_build_sext(Z3_ast expr, uint8_t bits) {
  return registerExpression(Z3_mk_sign_ext(g_context, bits, expr));
}

Z3_ast _sym_build_zext(Z3_ast expr, uint8_t bits) {
  if (expr == nullptr)
    return nullptr;
    
  return registerExpression(Z3_mk_zero_ext(g_context, bits, expr));
}

Z3_ast _sym_build_trunc(Z3_ast expr, uint8_t bits) {
  if (expr == nullptr)
    return nullptr;

  return registerExpression(Z3_mk_extract(g_context, bits - 1, 0, expr));
}

Z3_ast _sym_build_int_to_float(Z3_ast value, int is_double, int is_signed) {
  auto *sort = FSORT(is_double);
  Z3_inc_ref(g_context, (Z3_ast)sort);
  auto *result = registerExpression(
      is_signed
          ? Z3_mk_fpa_to_fp_signed(g_context, g_rounding_mode, value, sort)
          : Z3_mk_fpa_to_fp_unsigned(g_context, g_rounding_mode, value, sort));
  Z3_dec_ref(g_context, (Z3_ast)sort);
  return result;
}

Z3_ast _sym_build_float_to_float(Z3_ast expr, int to_double) {
  auto *sort = FSORT(to_double);
  Z3_inc_ref(g_context, (Z3_ast)sort);
  auto *result = registerExpression(
      Z3_mk_fpa_to_fp_float(g_context, g_rounding_mode, expr, sort));
  Z3_dec_ref(g_context, (Z3_ast)sort);
  return result;
}

Z3_ast _sym_build_bits_to_float(Z3_ast expr, int to_double) {
  if (expr == nullptr)
    return nullptr;

  auto *sort = FSORT(to_double);
  Z3_inc_ref(g_context, (Z3_ast)sort);
  auto *result = registerExpression(Z3_mk_fpa_to_fp_bv(g_context, expr, sort));
  Z3_dec_ref(g_context, (Z3_ast)sort);
  return result;
}

Z3_ast _sym_build_float_to_bits(Z3_ast expr) {
  if (expr == nullptr)
    return nullptr;
  return registerExpression(Z3_mk_fpa_to_ieee_bv(g_context, expr));
}

Z3_ast _sym_build_float_to_signed_integer(Z3_ast expr, uint8_t bits) {
  return registerExpression(Z3_mk_fpa_to_sbv(
      g_context, Z3_mk_fpa_round_toward_zero(g_context), expr, bits));
}

Z3_ast _sym_build_float_to_unsigned_integer(Z3_ast expr, uint8_t bits) {
  return registerExpression(Z3_mk_fpa_to_ubv(
      g_context, Z3_mk_fpa_round_toward_zero(g_context), expr, bits));
}

Z3_ast _sym_build_bool_to_bit(Z3_ast expr) {
  if (expr == nullptr)
    return nullptr;

  return registerExpression(Z3_mk_ite(g_context, expr, _sym_build_integer(1, 1),
                                      _sym_build_integer(0, 1)));
}

#if DEBUG_CHECK_INPUTS

std::vector<uint8_t> getConcreteValues(Z3_model model) {
  unsigned num_constants =  Z3_model_get_num_consts(g_context, model);
  std::vector<uint8_t> values = inputs_;
  for (unsigned i = 0; i < num_constants; i++) {

    Z3_func_decl decl = Z3_model_get_const_decl(g_context, model, i);
    Z3_ast e = Z3_model_get_const_interp(g_context, model, decl);
    // Z3_decl_kind kind = Z3_get_decl_kind(g_context, decl);
    Z3_symbol symbol = Z3_get_decl_name(g_context, decl);
    Z3_symbol_kind symbol_kind = Z3_get_symbol_kind(g_context, symbol);

    if (symbol_kind == Z3_STRING_SYMBOL) {
      uint64_t value;
      Z3_get_numeral_uint64(g_context, e, &value);
      const char* name = Z3_get_symbol_string(g_context, symbol);
      int idx = atoi(name + 5 /* "stdin" */);
      values[idx] = (uint8_t)value;
    }
  }
  return values;
}

static uint32_t debug_count;
static uint32_t debug_hash;
static uint8_t debug_taken;
void saveValues(std::vector<uint8_t>& values) {

  static char* out_dir = nullptr;
  if (out_dir == nullptr) {
    out_dir = getenv("SYMCC_OUTPUT_DIR");
    assert(out_dir);
  }
  std::string fname = std::string(out_dir) + "/input";
  static char s_count[16];
  static char s_hash[32];
  sprintf(s_hash, "%x", debug_hash);
  sprintf(s_count, "%d", debug_count);
  fname = fname + "_" + std::string(s_hash) + "_" + std::string(s_count) + "_" + (debug_taken ? "1" : "0");

  std::ofstream of(fname, std::ofstream::out | std::ofstream::binary);
  printf("New testcase: %s\n", fname.c_str());
  if (of.fail())
    printf("Unable to open a file to write results\n");

  // TODO: batch write
  for (unsigned i = 0; i < values.size(); i++) {
    char val = values[i];
    of.write(&val, sizeof(val));
  }

  of.close();
}

static uint64_t fuzz_check_count = 0;
void saveDebugInput(std::vector<uint8_t>& values, uint64_t debug_value) {

  static char* out_dir = nullptr;
  if (out_dir == nullptr) {
    out_dir = getenv("SYMCC_OUTPUT_DIR");
    assert(out_dir);
  }
  std::string fname = std::string(out_dir) + "/debug";
  static char s_count[16];
  static char s_value[32];
  sprintf(s_value, "%lx", debug_value);
  sprintf(s_count, "%05ld", fuzz_check_count);
  fname = fname + "_" + std::string(s_count) + "_" + std::string(s_value);

  std::ofstream of(fname, std::ofstream::out | std::ofstream::binary);
  printf("DEBUG testcase: %s\n", fname.c_str());
  if (of.fail())
    printf("Unable to open a file to write results\n");

  // TODO: batch write
  for (unsigned i = 0; i < values.size(); i++) {
    char val = values[i];
    of.write(&val, sizeof(val));
  }

  of.close();
}

int checkConsistencySMT(Z3_ast e, uint64_t expected_value);
#endif

void _sym_push_path_constraint(Z3_ast constraint, int taken,
                               uintptr_t site_id [[maybe_unused]]) {
  if (constraint == nullptr)
    return;

#if DEBUG_CHECK_INPUTS
  debug_count += 1;
  debug_hash = debug_hash ^ site_id;
  debug_taken = taken;

  static int check_input = -1;
  static uint32_t check_input_count = 0;
  static uint32_t check_input_hash = 0;
  static uint32_t check_input_taken = 0;
  if (check_input == -1) {

    if (getenv("DEBUG_CHECK_INPUT"))
      check_input = 1;
    else
      check_input = 0;

    if (check_input) {
      if (getenv("DEBUG_CHECK_INPUT_COUNT"))
        check_input_count = atoi(getenv("DEBUG_CHECK_INPUT_COUNT"));
      else
        abort();

      if (getenv("DEBUG_CHECK_INPUT_HASH"))
        check_input_hash = strtol(getenv("DEBUG_CHECK_INPUT_HASH"), NULL, 16);
      else
        abort();
      
      if (getenv("DEBUG_CHECK_INPUT_TAKEN"))
        check_input_taken = atoi(getenv("DEBUG_CHECK_INPUT_TAKEN"));
      else
        abort();
    }
  }

  if (check_input) {
    // printf("Checking...\n");
    if (debug_count == check_input_count) {
      if (debug_hash == check_input_hash) {
        if (debug_taken != check_input_taken) {
          printf("Input is taking the expected direction!\n");
          exit(0);
        } else {
          printf("Input is divergent: it reaches the same branch but does not take the expected direction!\n");
          exit(66);
        }
      } else {
        printf("Input is divergent: it does take the same path! [hash is different: %x vs expected=%x]\n", debug_hash, check_input_hash);
        exit(66);
      }
    } else if (debug_count > check_input_count) {
      printf("Input is divergent: it does take the same path! [count is larger]\n");
      exit(66);
    }
  } 
#endif

  constraint = Z3_simplify(g_context, constraint);
  Z3_inc_ref(g_context, constraint);

  /* Check the easy cases first: if simplification reduced the constraint to
     "true" or "false", there is no point in trying to solve the negation or *
     pushing the constraint to the solver... */

  if (Z3_is_eq_ast(g_context, constraint, Z3_mk_true(g_context))) {
    assert(taken && "We have taken an impossible branch");
    Z3_dec_ref(g_context, constraint);
    return;
  }

  if (Z3_is_eq_ast(g_context, constraint, Z3_mk_false(g_context))) {
    assert(!taken && "We have taken an impossible branch");
    Z3_dec_ref(g_context, constraint);
    return;
  }

  /* Generate a solution for the alternative */
  Z3_ast not_constraint =
      Z3_simplify(g_context, Z3_mk_not(g_context, constraint));
  Z3_inc_ref(g_context, not_constraint);

  Z3_solver_push(g_context, g_solver);
  Z3_solver_assert(g_context, g_solver, taken ? not_constraint : constraint);
  fprintf(g_log, "Trying to solve:\n%s\n",
          Z3_solver_to_string(g_context, g_solver));

  Z3_lbool feasible;
#if DEBUG_SKIP_QUERIES
  static int skip = -1;
  if (skip == -1) {
    if (getenv("SYMCC_SKIP_QUERIES"))
      skip = 1;
    else
      skip = 0;
  }
  if (skip)
    feasible = Z3_L_FALSE;
  else
#endif
  feasible = Z3_solver_check(g_context, g_solver);
  if (feasible == Z3_L_TRUE) {
    Z3_model model = Z3_solver_get_model(g_context, g_solver);
    Z3_model_inc_ref(g_context, model);
    fprintf(g_log, "Found diverging input:\n%s\n",
            Z3_model_to_string(g_context, model));

#if DEBUG_CHECK_INPUTS
    std::vector<uint8_t> values = getConcreteValues(model);
    saveValues(values);
#endif

    Z3_model_dec_ref(g_context, model);
  } else {
    fprintf(g_log, "Can't find a diverging input at this point\n");
  }
  fflush(g_log);

  Z3_solver_pop(g_context, g_solver, 1);

  /* Assert the actual path constraint */
  Z3_ast newConstraint = (taken ? constraint : not_constraint);
  Z3_inc_ref(g_context, newConstraint);
  Z3_solver_assert(g_context, g_solver, newConstraint);
  assert((Z3_solver_check(g_context, g_solver) == Z3_L_TRUE) &&
         "Asserting infeasible path constraint");
  Z3_dec_ref(g_context, constraint);
  Z3_dec_ref(g_context, not_constraint);

#if DEBUG_CHECK_PI_CONCRETE
  printf("Checking PI CONCRETE\n");
  Z3_ast_vector r = Z3_solver_get_assertions(g_context, g_solver);
  Z3_ast_vector_inc_ref(g_context, r);
  if (Z3_ast_vector_size(g_context, r) > 0) {
    Z3_ast* array = (Z3_ast*) malloc(sizeof(Z3_ast) * Z3_ast_vector_size(g_context, r));
    for(uint32_t i = 0; i < Z3_ast_vector_size(g_context, r); i++)
      array[i] = Z3_ast_vector_get(g_context, r, i);
    Z3_ast query = Z3_mk_and(g_context, Z3_ast_vector_size(g_context, r), array);
    if(checkConsistencySMT(query, 1) == 0) {
      printf("Adding infeasible constraints: %s\n", Z3_ast_to_string(g_context, (taken ? constraint : not_constraint)));
      abort();
    } else {
      // printf("\nPI OK\n\n");
    }
  }
  Z3_ast_vector_dec_ref(g_context, r);
#endif
}

SymExpr _sym_concat_helper(SymExpr a, SymExpr b) {
  return registerExpression(Z3_mk_concat(g_context, a, b));
}

SymExpr _sym_extract_helper(SymExpr expr, size_t first_bit, size_t last_bit) {
  return registerExpression(
      Z3_mk_extract(g_context, first_bit, last_bit, expr));
}

size_t _sym_bits_helper(SymExpr expr) {
  auto *sort = Z3_get_sort(g_context, expr);
  Z3_inc_ref(g_context, (Z3_ast)sort);
  auto result = Z3_get_bv_sort_size(g_context, sort);
  Z3_dec_ref(g_context, (Z3_ast)sort);
  return result;
}

/* No call-stack tracing */
void _sym_notify_call(uintptr_t) {}
void _sym_notify_ret(uintptr_t) {}
#if DEBUG_CONSISTENCY_CHECK
static uint64_t last_bb = 0;
void _sym_notify_basic_block(uintptr_t id) {
  last_bb = id;
}
#else
void _sym_notify_basic_block(uintptr_t) {}
#endif

/* Debugging */
const char *_sym_expr_to_string(SymExpr expr) {
  return Z3_ast_to_string(g_context, expr);
}

bool _sym_feasible(SymExpr expr) {
  expr = Z3_simplify(g_context, expr);
  Z3_inc_ref(g_context, expr);

  Z3_solver_push(g_context, g_solver);
  Z3_solver_assert(g_context, g_solver, expr);
  Z3_lbool feasible = Z3_solver_check(g_context, g_solver);
  Z3_solver_pop(g_context, g_solver, 1);

  Z3_dec_ref(g_context, expr);
  return (feasible == Z3_L_TRUE);
}

/* Garbage collection */
void _sym_collect_garbage() {
  if (allocatedExpressions.size() < g_config.garbageCollectionThreshold)
    return;

#ifndef NDEBUG
  auto start = std::chrono::high_resolution_clock::now();
  auto startSize = allocatedExpressions.size();
#endif

  auto reachableExpressions = collectReachableExpressions();
  for (auto expr_it = allocatedExpressions.begin();
       expr_it != allocatedExpressions.end();) {
    if (reachableExpressions.count(*expr_it) == 0) {
      expr_it = allocatedExpressions.erase(expr_it);
    } else {
      ++expr_it;
    }
  }

#ifndef NDEBUG
  auto end = std::chrono::high_resolution_clock::now();
  auto endSize = allocatedExpressions.size();

  std::cerr << "After garbage collection: " << endSize
            << " expressions remain (before: " << startSize << ")" << std::endl
            << "\t(collection took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " milliseconds)" << std::endl;
#endif
}

/* Test-case handling */
void symcc_set_test_case_handler(TestCaseHandler) {
  // The simple backend doesn't support test-case handlers. However, let's not
  // make this a fatal error; otherwise, users would have to change their
  // programs to make them work with the simple backend.
  fprintf(
      g_log,
      "Warning: test-case handlers aren't supported in the simple backend\n");
}

#if DEBUG_CONSISTENCY_CHECK

int checkConsistencySMT(Z3_ast e, uint64_t expected_value) {

  static Z3_model m = NULL;
  if (m == NULL) {
    Z3_set_ast_print_mode(g_context, Z3_PRINT_LOW_LEVEL);
    std::vector<uint8_t> values = inputs_;
    m = Z3_mk_model(g_context);
    Z3_model_inc_ref(g_context, m);
    Z3_sort sort = Z3_mk_bv_sort(g_context, 8);
    for (size_t i = 0; i < inputs_.size(); i++) {

      // Z3_symbol s =  Z3_mk_int_symbol(g_context, i);
      Z3_ast v = Z3_mk_int(g_context, inputs_[i], sort);
      Z3_inc_ref(g_context, v);

      // printf("input[%ld] = %x\n", i, inputs_[i]);
      // printf("%s\n", Z3_ast_to_string(g_context, v));

      auto varName = "stdin" + std::to_string(i);
      Z3_symbol s = Z3_mk_string_symbol(g_context, varName.c_str());

      Z3_func_decl decl = Z3_mk_func_decl(g_context, s, 0, NULL, sort);
      Z3_add_const_interp(g_context, m, decl, v);
    }

    // printf("Model:\n%s\n", Z3_model_to_string(g_context, m));
  }

  // printf("EXPR: %s\n", Z3_ast_to_string(g_context, e));

  uint64_t  value;
  Z3_ast    solution = nullptr;
  Z3_bool   successfulEval =
      Z3_model_eval(g_context, m, e, Z3_TRUE, &solution);
  assert(successfulEval && "Failed to evaluate model");
  if (!successfulEval) abort();

  if (Z3_get_ast_kind(g_context, solution) == Z3_NUMERAL_AST) {
    Z3_bool successGet =
          Z3_get_numeral_uint64(g_context, solution, (uint64_t*)&value);
    assert(successGet);
    if (value != expected_value) {
      Z3_set_ast_print_mode(g_context, Z3_PRINT_LOW_LEVEL);
      printf("[%d] %s\n", successGet, Z3_ast_to_string(g_context, e));
      printf("FAILURE: %lx vs expected=%lx\n", value, expected_value);
    } else {
      printf("SUCCESS: %lx vs expected=%lx\n", value, expected_value);
    }
    return value == expected_value;
  } else {

    Z3_lbool res = Z3_get_bool_value(g_context, solution);
    if (res == Z3_L_TRUE) {
      value = 1;
      if (value != expected_value) {
        Z3_set_ast_print_mode(g_context, Z3_PRINT_LOW_LEVEL);
        printf("%s\n", Z3_ast_to_string(g_context, e));
        printf("BOOL FAILURE: %lx vs expected=%lx\n", value, expected_value);
      }
      return value == expected_value;
    } else if (res == Z3_L_FALSE) {
      value = 0;
      if (value != expected_value) {
        Z3_set_ast_print_mode(g_context, Z3_PRINT_LOW_LEVEL);
        printf("%s\n", Z3_ast_to_string(g_context, e));
        printf("BOOL FAILURE: %lx vs expected=%lx\n", value, expected_value);
      }
      return value == expected_value;
    } else {
      printf("KIND: %x\n", Z3_get_ast_kind(g_context, solution));
      Z3_set_ast_print_mode(g_context, Z3_PRINT_LOW_LEVEL);
      printf("EXPR: %s\n", Z3_ast_to_string(g_context, e));
      assert(0 && "Cannot evaluate");
      abort();
    }
  }
}

int checkConsistency(Z3_ast e, uint64_t expected_value) {
  return checkConsistencySMT(e, expected_value);
}

void _sym_check_consistency(SymExpr expr, uint64_t expected_value, uint64_t) {
  if (expr == NULL) return;
  int res = checkConsistency(expr, expected_value);
  if (res == 0) {
    printf("CONSISTENCY CHECK FAILED AT %lx\n", last_bb);
    abort();
  }

#if DEBUG_FUZZ_EXPRS

  int fuzz_expr = -1;
  uint64_t fuzz_count = 0;
  uint64_t fuzz_value = 0;
  if (fuzz_expr == -1) {

    if (getenv("DEBUG_FUZZ_EXPR"))
      fuzz_expr = 1;
    else
      fuzz_expr = 0;

    if (fuzz_expr) {
      if (getenv("DEBUG_FUZZ_EXPR_COUNT"))
        fuzz_count = atoi(getenv("DEBUG_FUZZ_EXPR_COUNT"));
      else
        abort();

      if (getenv("DEBUG_FUZZ_EXPR_VALUE"))
        fuzz_value = strtol(getenv("DEBUG_FUZZ_EXPR_VALUE"), NULL, 16);
      else
        abort();
    }
  }

  fuzz_check_count += 1;
  if (fuzz_expr) {
    if (fuzz_count == fuzz_check_count) {
      if (fuzz_value == expected_value) {
        printf("Expression has expected value!\n");
        exit(0);
      } else {
        printf("Expression has wrong value [%lx vs expected=%lx]\n", expected_value, fuzz_value);
        exit(66);
      }
    } else if (fuzz_count < fuzz_check_count) {
      printf("Expression check has been bypassed [count is larger]\n");
      exit(66);
    }
    return;
  }

  Z3_solver_push(g_context, g_solver);
  Z3_sort sort = Z3_get_sort(g_context, expr);
  Z3_ast not_e;
  if (Z3_get_sort_kind(g_context, sort) == Z3_BOOL_SORT)
    not_e = Z3_mk_not(g_context, expr);
  else
    not_e = _sym_build_not_equal(expr, _sym_build_integer(expected_value, _sym_bits_helper(expr)));
  Z3_solver_assert(g_context, g_solver, not_e);
  Z3_lbool feasible = Z3_solver_check(g_context, g_solver);
  if (feasible == Z3_L_TRUE) {
    Z3_model model = Z3_solver_get_model(g_context, g_solver);
    Z3_model_inc_ref(g_context, model);
    std::vector<uint8_t> values = getConcreteValues(model);
    Z3_ast solution = nullptr;
    Z3_model_eval(g_context, model, expr, Z3_TRUE, &solution);
    uint64_t value = 0;
    if (Z3_get_sort_kind(g_context, sort) == Z3_BOOL_SORT) {
      Z3_lbool res = Z3_get_bool_value(g_context, solution);
      if (res == Z3_L_TRUE)
        value = 1;
      else if (res == Z3_L_FALSE)
        value = 0;
      else
        abort();
    } else
      Z3_get_numeral_uint64(g_context, solution, &value);
    saveDebugInput(values, value);
    Z3_model_dec_ref(g_context, model);
  } 
  Z3_solver_pop(g_context, g_solver, 1);
#endif
}
#endif