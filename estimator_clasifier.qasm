OPENQASM 2.0;
include "qelib1.inc";
gate gate_RealAmplitudes(param0,param1,param2,param3,param4,param5,param6,param7,param8,param9,param10,param11,param12,param13,param14,param15) q0,q1,q2,q3 { ry(1.22670192) q0; ry(1.56641047) q1; ry(1.79906141) q2; ry(1.5357788) q3; cx q2,q3; cx q1,q2; cx q0,q1; ry(1.19234226) q0; ry(1.47601814) q1; ry(2.32606606) q2; ry(1.667176) q3; cx q2,q3; cx q1,q2; cx q0,q1; ry(0.524446485) q0; ry(0.188686986) q1; ry(0.475849645) q2; ry(1.46071512) q3; cx q2,q3; cx q1,q2; cx q0,q1; ry(0.0645662288) q0; ry(-0.0015609432) q1; ry(0.0709442986) q2; ry(0.917373156) q3; }
qreg q[4];
gate_RealAmplitudes(1.22670192,1.56641047,1.79906141,1.5357788,1.19234226,1.47601814,2.32606606,1.667176,0.524446485,0.188686986,0.475849645,1.46071512,0.0645662288,-0.0015609432,0.0709442986,0.917373156) q[0],q[1],q[2],q[3];
