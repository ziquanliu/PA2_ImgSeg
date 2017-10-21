#include<math.h>
#define E 2.718281823
double cal_kernel(double* x,double* x_old,double hp,double hc){
  double left_t,left;
  double right_t,right;
  left_t=(x[0]-x_old[0])*(x[0]-x_old[0]) + (x[1]-x_old[1])*(x[1]-x_old[1]);
  right_t=(x[2]-x_old[2])*(x[2]-x_old[2]) + (x[3]-x_old[3])*(x[3]-x_old[3]);
  left=-left_t*0.5/(hc*hc);
  right=-right_t*0.5/(hp*hp);
  return pow(E,left+right);
}
