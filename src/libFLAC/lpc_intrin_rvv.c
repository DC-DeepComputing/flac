/* libFLAC - Free Lossless Audio Codec library
 * Copyright (C) 2000-2009  Josh Coalson
 * Copyright (C) 2011-2025  Xiph.Org Foundation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * - Neither the name of the Xiph.org Foundation nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include "private/lpc.h"
#include "private/bitmath.h"
#include "share/compat.h"
#include "private/macros.h"
#include <math.h>
#include <riscv_vector.h>

/* RISC-V RVV 1.0 (VLEN >= 128) intrinsics for the encoder LPC hot kernels.
   This file is compiled with -march=rv64gcv; all code is guarded so the
   file is empty on non-RISC-V targets.

   Note on numerics: like the upstream NEON/AVX2 paths, the RVV
   autocorrelation accumulates through vector lanes, which may differ from
   the scalar path in the last ulps of the double sums.  FLAC guarantees
   that different CPU feature paths produce equally lossless output; the
   decoded audio is unaffected.

   Integer kernels are exactly bit-exact with the portable C:
   - residual 32-bit relies on two's-complement wraparound (mod 2^32
     addition is associative),
   - the wide variant is NOT dispatched (measured slower than scalar C on
     SpacemiT X100; kept here for reference). */

#if defined(__riscv) && (__riscv_xlen == 64) && defined(__riscv_vector)

#define ACC_DECL(N)  vfloat64m2_t acc##N = __riscv_vfmv_v_f_f64m2(0.0, 8);
#define ACC_FMA(N)   acc##N = __riscv_vfmacc_vv_f64m2(acc##N, d64, __riscv_vfwcvt_f_f_v_f64m2(__riscv_vle32_v_f32m1(&data[i - N], 8), 8), 8);
#define ACC_RED(N)   autoc[N] += __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(acc##N, zero, 8));

#define AUTOC_CHUNK(NAMES, NDECL, NFMA, NRED) do { \
	NAMES(NDECL) \
	uint32_t i, j; \
	vfloat64m1_t zero = __riscv_vfmv_s_f_f64m1(0.0, 8); \
	/* head: scalar triangular part, ascending sample order */ \
	for (i = 0; i < MAX_LAG && i < data_len; i++) { \
		const double d = (double)data[i]; \
		for (j = 0; j <= i; j++) autoc[j] += d * (double)data[i - j]; \
	} \
	/* main: 8 samples per iteration, one accumulator vector per lag */ \
	for (; i + 8 <= data_len; i += 8) { \
		vfloat32m1_t d32 = __riscv_vle32_v_f32m1(&data[i], 8); \
		vfloat64m2_t d64 = __riscv_vfwcvt_f_f_v_f64m2(d32, 8); \
		NAMES(NFMA) \
	} \
	/* tail: scalar, ascending sample order */ \
	for (; i < data_len; i++) { \
		const double d = (double)data[i]; \
		for (j = 0; j < MAX_LAG; j++) autoc[j] += d * (double)data[i - j]; \
	} \
	NAMES(NRED) \
} while(0)

#define L8(x)   x(0) x(1) x(2) x(3) x(4) x(5) x(6) x(7)
#define L12(x)  L8(x) x(8) x(9) x(10) x(11)
#define L16(x)  L12(x) x(12) x(13) x(14) x(15)

void FLAC__lpc_compute_autocorrelation_rvv(const FLAC__real data[], uint32_t data_len, uint32_t lag, double autoc[])
{
	if (data_len >= 64) {
		uint32_t MAX_LAG;
		if (lag <= 8)       { MAX_LAG = 8;  AUTOC_CHUNK(L8,  ACC_DECL, ACC_FMA, ACC_RED); return; }
		else if (lag <= 12) { MAX_LAG = 12; AUTOC_CHUNK(L12, ACC_DECL, ACC_FMA, ACC_RED); return; }
		else if (lag <= 16) { MAX_LAG = 16; AUTOC_CHUNK(L16, ACC_DECL, ACC_FMA, ACC_RED); return; }
	}
	{
		/* generic fallback for short blocks / large lags: one vector
		   across the lags, per sample. */
		uint32_t i, j;
		for (j = 0; j < lag; j++) autoc[j] = 0.0;
		for (i = 0; i < lag && i < data_len; i++) {
			const double d = (double)data[i];
			for (j = 0; j <= i; j++) autoc[j] += d * (double)data[i - j];
		}
		if (data_len > lag) {
			size_t vl = lag;
			double tmp[32];
			for (j = 0; j < lag; j++) tmp[j] = autoc[lag - 1 - j];
			vfloat64m8_t accw = __riscv_vle64_v_f64m8(tmp, vl);
			for (i = lag; i < data_len; i++) {
				const double d = (double)data[i];
				vfloat32m4_t w32 = __riscv_vle32_v_f32m4(&data[i + 1 - lag], vl);
				vfloat64m8_t w = __riscv_vfwcvt_f_f_v_f64m8(w32, vl);
				accw = __riscv_vfmacc_vf_f64m8(accw, d, w, vl);
			}
			vfloat64m8_t rev = __riscv_vrgather_vv_f64m8(accw,
				__riscv_vrsub_vx_u64m8(__riscv_vid_v_u64m8(vl), (uint64_t)(lag - 1), vl), vl);
			__riscv_vse64_v_f64m8(autoc, rev, vl);
		}
	}
}

/* m1 dual-accumulator kernel kept for reference; the dispatched kernel
   below (m4 sample-parallel) is ~2x faster at VLEN=256. */
static void FLAC__lpc_residual_m1(const FLAC__int32 data[], uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[])
{
	uint32_t i = 0;
	const size_t vl = 8;

	for (; i + 8 <= data_len; i += 8) {
		vint32m1_t acc0 = __riscv_vmv_v_x_i32m1(0, vl);
		vint32m1_t acc1 = __riscv_vmv_v_x_i32m1(0, vl);
		uint32_t half = order >> 1, j;
		for (j = 0; j < half; j++) {
			vint32m1_t d = __riscv_vle32_v_i32m1(&data[(int)i - 1 - (int)j], vl);
			acc0 = __riscv_vmacc_vx_i32m1(acc0, qlp_coeff[j], d, vl);
		}
		for (j = half; j < order; j++) {
			vint32m1_t d = __riscv_vle32_v_i32m1(&data[(int)i - 1 - (int)j], vl);
			acc1 = __riscv_vmacc_vx_i32m1(acc1, qlp_coeff[j], d, vl);
		}
		vint32m1_t acc = __riscv_vadd_vv_i32m1(acc0, acc1, vl);
		acc = __riscv_vsra_vx_i32m1(acc, lp_quantization, vl);
		vint32m1_t dv = __riscv_vle32_v_i32m1(&data[i], vl);
		vint32m1_t res = __riscv_vsub_vv_i32m1(dv, acc, vl);
		__riscv_vse32_v_i32m1(&residual[i], res, vl);
	}
	for (; i < data_len; i++) {
		FLAC__int32 sum = 0;
		uint32_t j;
		for (j = 0; j < order; j++)
			sum += qlp_coeff[j] * data[(int)i - 1 - (int)j];
		residual[i] = data[i] - (sum >> lp_quantization);
	}
}

/* Sample-parallel i32m4 kernel: each iteration computes `vl` consecutive
   residual samples at once. vsetvl folds the tail into the vector loop,
   so non-8-multiple lengths (15/63/127/...) stay fast. The encoder calls
   us with `signal+order`, so data[-1..-order] is a readable history area.
   mod-2^32 addition is associative => bit-exact with portable C. */
void FLAC__lpc_compute_residual_from_qlp_coefficients_rvv(const FLAC__int32 data[], uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[])
{
	uint32_t i = 0;

	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

	while (i < data_len) {
		size_t vl = __riscv_vsetvl_e32m4(data_len - i);
		vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
		uint32_t j;
		for (j = 0; j < order; j++) {
			vint32m4_t d = __riscv_vle32_v_i32m4(&data[(int)i - 1 - (int)j], vl);
			acc = __riscv_vmacc_vx_i32m4(acc, qlp_coeff[j], d, vl);
		}
		acc = __riscv_vsra_vx_i32m4(acc, lp_quantization, vl);
		vint32m4_t dv = __riscv_vle32_v_i32m4(&data[i], vl);
		__riscv_vse32_v_i32m4(&residual[i], __riscv_vsub_vv_i32m4(dv, acc, vl), vl);
		i += vl;
	}
}

void FLAC__lpc_compute_residual_from_qlp_coefficients_wide_rvv(const FLAC__int32 data[], uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[])
{
	/* Reference implementation: measured slower than the portable scalar
	   version on SpacemiT X100 (64-bit vector MACs are half-rate there),
	   so it is not dispatched by the encoder.  Kept for future tuning. */
	uint32_t i = 0;
	const size_t vl = 8;

	for (; i + 8 <= data_len; i += 8) {
		vint64m2_t acc0 = __riscv_vmv_v_x_i64m2(0, vl);
		vint64m2_t acc1 = __riscv_vmv_v_x_i64m2(0, vl);
		uint32_t half = order >> 1, j;
		for (j = 0; j < half; j++) {
			vint32m1_t d32 = __riscv_vle32_v_i32m1(&data[(int)i - 1 - (int)j], vl);
			vint64m2_t d64 = __riscv_vsext_vf2_i64m2(d32, vl);
			acc0 = __riscv_vmacc_vx_i64m2(acc0, qlp_coeff[j], d64, vl);
		}
		for (j = half; j < order; j++) {
			vint32m1_t d32 = __riscv_vle32_v_i32m1(&data[(int)i - 1 - (int)j], vl);
			vint64m2_t d64 = __riscv_vsext_vf2_i64m2(d32, vl);
			acc1 = __riscv_vmacc_vx_i64m2(acc1, qlp_coeff[j], d64, vl);
		}
		vint64m2_t acc = __riscv_vadd_vv_i64m2(acc0, acc1, vl);
		acc = __riscv_vsra_vx_i64m2(acc, lp_quantization, vl);
		vint32m1_t dv32 = __riscv_vle32_v_i32m1(&data[i], vl);
		vint64m2_t dv = __riscv_vsext_vf2_i64m2(dv32, vl);
		vint64m2_t res = __riscv_vsub_vv_i64m2(dv, acc, vl);
		vint32m1_t resn = __riscv_vnsra_wx_i32m1(res, 0, vl);
		__riscv_vse32_v_i32m1(&residual[i], resn, vl);
	}
	for (; i < data_len; i++) {
		FLAC__int64 sum = 0;
		uint32_t j;
		for (j = 0; j < order; j++)
			sum += (FLAC__int64)qlp_coeff[j] * (FLAC__int64)data[(int)i - 1 - (int)j];
		residual[i] = data[i] - (FLAC__int32)(sum >> lp_quantization);
	}
}


/* ---- fixed predictor + partition sums (integer, bit-exact with portable) ---- */

/* signed abs matching local_abs(): INT_MIN stays INT_MIN */
#define VABS_I32(v) __riscv_vmax_vv_i32m1((v), __riscv_vneg_v_i32m1(v, vl), vl)

uint32_t FLAC__fixed_compute_best_predictor_wide_rvv(const FLAC__int32 data[], uint32_t data_len, float residual_bits_per_sample[FLAC__MAX_FIXED_ORDER+1])
{
	const size_t vl = 8;
	vuint64m2_t te0 = __riscv_vmv_v_x_u64m2(0, vl);
	vuint64m2_t te1 = __riscv_vmv_v_x_u64m2(0, vl);
	vuint64m2_t te2 = __riscv_vmv_v_x_u64m2(0, vl);
	vuint64m2_t te3 = __riscv_vmv_v_x_u64m2(0, vl);
	vuint64m2_t te4 = __riscv_vmv_v_x_u64m2(0, vl);
	FLAC__uint64 se0 = 0, se1 = 0, se2 = 0, se3 = 0, se4 = 0;
	int i = 0;

	for (; i + 8 <= (int)data_len; i += 8) {
		vint32m1_t d0 = __riscv_vle32_v_i32m1(&data[i], vl);
		vint32m1_t d1 = __riscv_vle32_v_i32m1(&data[i-1], vl);
		vint32m1_t d2 = __riscv_vle32_v_i32m1(&data[i-2], vl);
		vint32m1_t d3 = __riscv_vle32_v_i32m1(&data[i-3], vl);
		vint32m1_t d4 = __riscv_vle32_v_i32m1(&data[i-4], vl);
		vint32m1_t e;
		e = d0;
		te0 = __riscv_vadd_vv_u64m2(te0, __riscv_vreinterpret_v_i64m2_u64m2(__riscv_vsext_vf2_i64m2(VABS_I32(e), vl)), vl);
		e = __riscv_vsub_vv_i32m1(d0, d1, vl);
		te1 = __riscv_vadd_vv_u64m2(te1, __riscv_vreinterpret_v_i64m2_u64m2(__riscv_vsext_vf2_i64m2(VABS_I32(e), vl)), vl);
		e = __riscv_vadd_vv_i32m1(__riscv_vnmsac_vx_i32m1(d0, 2, d1, vl), d2, vl);
		te2 = __riscv_vadd_vv_u64m2(te2, __riscv_vreinterpret_v_i64m2_u64m2(__riscv_vsext_vf2_i64m2(VABS_I32(e), vl)), vl);
		{
			vint32m1_t a = __riscv_vnmsac_vx_i32m1(d0, 3, d1, vl);
			e = __riscv_vsub_vv_i32m1(__riscv_vmacc_vx_i32m1(a, 3, d2, vl), d3, vl);
		}
		te3 = __riscv_vadd_vv_u64m2(te3, __riscv_vreinterpret_v_i64m2_u64m2(__riscv_vsext_vf2_i64m2(VABS_I32(e), vl)), vl);
		{
			vint32m1_t a = __riscv_vnmsac_vx_i32m1(d0, 4, d1, vl);
			a = __riscv_vmacc_vx_i32m1(a, 6, d2, vl);
			a = __riscv_vnmsac_vx_i32m1(a, 4, d3, vl);
			e = __riscv_vadd_vv_i32m1(a, d4, vl);
		}
		te4 = __riscv_vadd_vv_u64m2(te4, __riscv_vreinterpret_v_i64m2_u64m2(__riscv_vsext_vf2_i64m2(VABS_I32(e), vl)), vl);
	}
	/* tail: scalar with the exact portable semantics */
	for (; i < (int)data_len; i++) {
		FLAC__int32 e0 = data[i];
		FLAC__int32 e1 = data[i] - data[i-1];
		FLAC__int32 e2 = data[i] - 2*data[i-1] + data[i-2];
		FLAC__int32 e3 = data[i] - 3*data[i-1] + 3*data[i-2] - data[i-3];
		FLAC__int32 e4 = data[i] - 4*data[i-1] + 6*data[i-2] - 4*data[i-3] + data[i-4];
		se0 += (FLAC__uint64)(FLAC__int64)(e0<0?-e0:e0);
		se1 += (FLAC__uint64)(FLAC__int64)(e1<0?-e1:e1);
		se2 += (FLAC__uint64)(FLAC__int64)(e2<0?-e2:e2);
		se3 += (FLAC__uint64)(FLAC__int64)(e3<0?-e3:e3);
		se4 += (FLAC__uint64)(FLAC__int64)(e4<0?-e4:e4);
	}
	{
		vuint64m1_t zero = __riscv_vmv_v_x_u64m1(0, vl);
		FLAC__uint64 te[5];
		te[0]=__riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(te0, zero, vl)) + se0;
		te[1]=__riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(te1, zero, vl)) + se1;
		te[2]=__riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(te2, zero, vl)) + se2;
		te[3]=__riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(te3, zero, vl)) + se3;
		te[4]=__riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(te4, zero, vl)) + se4;

		FLAC__uint64 total_error_0=te[0], total_error_1=te[1], total_error_2=te[2], total_error_3=te[3], total_error_4=te[4];
		uint32_t order;

		if(total_error_0 <= flac_min(flac_min(flac_min(total_error_1, total_error_2), total_error_3), total_error_4))
			order = 0;
		else if(total_error_1 <= flac_min(flac_min(total_error_2, total_error_3), total_error_4))
			order = 1;
		else if(total_error_2 <= flac_min(total_error_3, total_error_4))
			order = 2;
		else if(total_error_3 <= total_error_4)
			order = 3;
		else
			order = 4;

		FLAC__ASSERT(data_len > 0 || total_error_0 == 0);
		residual_bits_per_sample[0] = (float)((total_error_0 > 0) ? log(M_LN2 * (double)total_error_0 / (double)data_len) / M_LN2 : 0.0);
		residual_bits_per_sample[1] = (float)((total_error_1 > 0) ? log(M_LN2 * (double)total_error_1 / (double)data_len) / M_LN2 : 0.0);
		residual_bits_per_sample[2] = (float)((total_error_2 > 0) ? log(M_LN2 * (double)total_error_2 / (double)data_len) / M_LN2 : 0.0);
		residual_bits_per_sample[3] = (float)((total_error_3 > 0) ? log(M_LN2 * (double)total_error_3 / (double)data_len) / M_LN2 : 0.0);
		residual_bits_per_sample[4] = (float)((total_error_4 > 0) ? log(M_LN2 * (double)total_error_4 / (double)data_len) / M_LN2 : 0.0);
		return order;
	}
}

void FLAC__precompute_partition_info_sums_rvv(const FLAC__int32 residual[], FLAC__uint64 abs_residual_partition_sums[],
	uint32_t residual_samples, uint32_t predictor_order, uint32_t min_partition_order, uint32_t max_partition_order, uint32_t bps)
{
	(void)bps;
	const uint32_t default_partition_samples = (residual_samples + predictor_order) >> max_partition_order;
	uint32_t partitions = 1u << max_partition_order;
	const size_t vl = 8;

	{
		/* walk identical to the portable version: the first partition
		   absorbs the predictor_order warmup entries (zero in the encoder). */
		uint32_t partition, residual_sample = 0, end = (uint32_t)(-(int)predictor_order);
		for(partition = 0; partition < partitions; partition++) {
			vuint64m2_t acc = __riscv_vmv_v_x_u64m2(0, vl);
			end += default_partition_samples;
			for( ; residual_sample + 8 <= end; residual_sample += 8) {
				vint32m1_t r = __riscv_vle32_v_i32m1(&residual[residual_sample], vl);
				vuint64m2_t a64 = __riscv_vzext_vf2_u64m2(__riscv_vreinterpret_v_i32m1_u32m1(VABS_I32(r)), vl);
				acc = __riscv_vadd_vv_u64m2(acc, a64, vl);
			}
			FLAC__uint64 s = 0;
			for( ; residual_sample < end; residual_sample++) {
				FLAC__int32 v = residual[residual_sample];
				s += (FLAC__uint64)(FLAC__int64)(v<0?-v:v);
			}
			FLAC__uint64 tot = s + __riscv_vmv_x_s_u64m1_u64(
				__riscv_vredsum_vs_u64m2_u64m1(acc, __riscv_vmv_v_x_u64m1(0, vl), vl));
			abs_residual_partition_sums[partition] = tot;
		}
	}

	{
		uint32_t from_partition = 0, to_partition = partitions;
		int partition_order;
		for(partition_order = (int)max_partition_order - 1; partition_order >= (int)min_partition_order; partition_order--) {
			uint32_t i;
			partitions >>= 1;
			for(i = 0; i < partitions; i++) {
				abs_residual_partition_sums[to_partition++] =
					abs_residual_partition_sums[from_partition  ] +
					abs_residual_partition_sums[from_partition+1];
				from_partition += 2;
			}
		}
	}
}

#endif /* __riscv */
