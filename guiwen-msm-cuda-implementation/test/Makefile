msma:
	nvcc -arch=sm_35 -rdc=true --expt-extended-lambda ./MSMtestbn.cu libgmp.a ../depends/libff-cuda/curves/alt_bn128/alt_bn128_pp_host.cu  ../depends/libff-cuda/curves/alt_bn128/alt_bn128_init_host.cu  ../depends/libff-cuda/curves/alt_bn128/alt_bn128_g1_host.cu ../depends/libff-cuda/curves/alt_bn128/alt_bn128_g2_host.cu  ../depends/libstl-cuda/memory.cu ../depends/libff-cuda/common/utils.cu ../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cu -o msmtesta
msmb:
	nvcc -arch=sm_35 -rdc=true --expt-extended-lambda ./MSMtestbls.cu libgmp.a ../depends/libff-cuda/curves/bls12_381/bls12_381_pp_host.cu  ../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cu  ../depends/libff-cuda/curves/bls12_381/bls12_381_g1_host.cu ../depends/libff-cuda/curves/bls12_381/bls12_381_g2_host.cu  ../depends/libstl-cuda/memory.cu ../depends/libff-cuda/common/utils.cu ../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cu -o msmtestb
msmm:
	nvcc -arch=sm_35 -rdc=true --expt-extended-lambda ./MSMtestmnt.cu libgmp.a ../depends/libff-cuda/curves/mnt4753/mnt4753_pp_host.cu ../depends/libff-cuda/curves/mnt4753/mnt4753_init_host.cu  ../depends/libff-cuda/curves/mnt4753/mnt4753_g1_host.cu ../depends/libff-cuda/curves/mnt4753/mnt4753_g2_host.cu ../depends/libstl-cuda/memory.cu ../depends/libff-cuda/common/utils.cu ../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cu -o msmtestm
gmsm:
	nvcc -arch=sm_86 -rdc=true --expt-extended-lambda ./guiwen-msm.cu libgmp.a ../depends/libff-cuda/curves/bls12_381/bls12_381_pp_host.cu  ../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cu  ../depends/libff-cuda/curves/bls12_381/bls12_381_g1_host.cu ../depends/libff-cuda/curves/bls12_381/bls12_381_g2_host.cu ../depends/libstl-cuda/set.cu ../depends/libstl-cuda/memory.cu ../depends/libff-cuda/common/utils.cu ../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cu -o gmsm

a:
	nvcc -arch=sm_35 -rdc=true --expt-extended-lambda ./testbn.cu libgmp.a ../depends/libff-cuda/curves/alt_bn128/alt_bn128_pp_host.cu  ../depends/libff-cuda/curves/alt_bn128/alt_bn128_init_host.cu  ../depends/libff-cuda/curves/alt_bn128/alt_bn128_g1_host.cu ../depends/libff-cuda/curves/alt_bn128/alt_bn128_g2_host.cu  ../depends/libstl-cuda/memory.cu ../depends/libff-cuda/common/utils.cu ../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cu   -o testa
b:
	nvcc -arch=sm_35 -rdc=true --expt-extended-lambda ./testbls.cu libgmp.a ../depends/libff-cuda/curves/bls12_381/bls12_381_pp_host.cu  ../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cu  ../depends/libff-cuda/curves/bls12_381/bls12_381_g1_host.cu ../depends/libff-cuda/curves/bls12_381/bls12_381_g2_host.cu  ../depends/libstl-cuda/memory.cu ../depends/libstl-cuda/io.cu ../depends/libff-cuda/common/utils.cu ../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cu   -o testb
m:
	nvcc -arch=sm_35 -rdc=true --expt-extended-lambda ./testmnt.cu libgmp.a ../depends/libff-cuda/curves/mnt4/mnt4_pp_host.cu  ../depends/libff-cuda/curves/mnt4/mnt4_init_host.cu  ../depends/libff-cuda/curves/mnt4/mnt4_g1_host.cu ../depends/libff-cuda/curves/mnt4/mnt4_g2_host.cu  ../depends/libstl-cuda/memory.cu ../depends/libff-cuda/common/utils.cu ../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cu   -o testm

tda:
	nvcc -arch=sm_86 -rdc=true --expt-extended-lambda ./test-double-addition.cu libgmp.a ../depends/libff-cuda/curves/bls12_381/bls12_381_pp_host.cu  ../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cu  ../depends/libff-cuda/curves/bls12_381/bls12_381_g1_host.cu ../depends/libff-cuda/curves/bls12_381/bls12_381_g2_host.cu  ../depends/libstl-cuda/memory.cu ../depends/libff-cuda/common/utils.cu  ../depends/libff-cuda/mini-mp-cuda/mini-mp-cuda.cu -o tda

