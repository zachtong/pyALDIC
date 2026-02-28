// Fast nearest, bi-linear and bi-cubic interpolation for image data
//
// Usage:
// ------
//     Z = ba_interp2(F, X, Y, [method])
//
// where method is one off nearest, linear, or cubic.
//
// F    is a WxHxD Image with an arbitray number of layers D.
// X, Y are I_1 x ... x I_n matrices with the x and y coordinates to
//      interpolate.
// Z    is a I_1 x ... x I_n x D matrix, which contains the interpolated image channels.
//
// Notes:
// ------
// This method handles the border by repeating the closest values to the point accessed. 
// This is different from matlabs border handling.
//
// Example
// ------
//
//    //// The mandrills eye
//    clear
//    IMG=load('mandrill');
//    IMG = ind2rgb(IMG.X, IMG.map);
//    [Dx Dy] = meshgrid(130:0.1:250, -150:0.1:-50);
//    
//    R = [cos(pi/4) sin(pi/4); -sin(pi/4) cos(pi/4)];
//    RD = R * [Dx(:)'; Dy(:)'] + 250;
//    RDx = reshape(RD(1,:), size(Dx));
//    RDy = reshape(RD(2,:), size(Dy));
//    
//    methods = {'nearest', 'linear', 'cubic'};
//    la=nan(1,3);
//    for i=1:3
//      la(i) = subplot(2,2,i);
//      tic;
//      IMG_R = ba_interp2(IMG, RDx, RDy, methods{i});
//      elapsed=toc;
//      imshow(IMG_R);
//      title(sprintf('Rotation and zoom using %s interpolation took %gs', methods{i}, elapsed));
//    end
//    linkaxes(la);
//
// Licence:
// --------
// GPL
// (c) 2008 Brian Amberg
// http://www.brian-amberg.de/

// Modified by Zixiang Tong & Jin Yang, 04/2024, bi-cubic Bspline interpolation
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
#include <mex.h>
// #include <stdlib.h>
#include <math.h>
// #include <string.h>
// #include <iostream>

inline 
static
int access(int M, int N, int x, int y) {
  if (x<0) x=0; else if (x>=N) x=N-1;
  if (y<0) y=0; else if (y>=M) y=M-1;
  return M*x + y;
}
 
inline
static
void indices_cubic_spline(
    int &f00_i,
    int &f10_i,
    int &f20_i,
    int &f30_i,
    int &f01_i,
    int &f11_i,
    int &f21_i,
    int &f31_i,
    int &f02_i,
    int &f12_i,
    int &f22_i,
    int &f32_i,
    int &f03_i,
    int &f13_i,
    int &f23_i,
    int &f33_i,
    const int x, const int y, 
    const mwSize &M, const mwSize &N) {
  if (x<=2 || y<=2 || x>=N-3 || y>=M-3) {
    f00_i = access(M, N, x-1, y-1);
    f10_i = access(M, N, x  , y-1);
    f20_i = access(M, N, x+1, y-1);
    f30_i = access(M, N, x+2, y-1);

    f01_i = access(M, N, x-1, y  );
    f11_i = access(M, N, x  , y  );
    f21_i = access(M, N, x+1, y  );
    f31_i = access(M, N, x+2, y  );

    f02_i = access(M, N, x-1, y+1);
    f12_i = access(M, N, x  , y+1);
    f22_i = access(M, N, x+1, y+1);
    f32_i = access(M, N, x+2, y+1);

    f03_i = access(M, N, x-1, y+2);
    f13_i = access(M, N, x  , y+2);
    f23_i = access(M, N, x+1, y+2);
    f33_i = access(M, N, x+2, y+2);
  } else {
    f00_i = access(M, N, x-1, y-1);
    f01_i = f00_i + 1;
    f02_i = f01_i + 1;
    f03_i = f02_i + 1;

    f10_i = f00_i + M;
    f11_i = f10_i + 1;
    f12_i = f11_i + 1;
    f13_i = f12_i + 1;

    f20_i = f10_i + M;
    f21_i = f20_i + 1;
    f22_i = f21_i + 1;
    f23_i = f22_i + 1;

    f30_i = f20_i + M;
    f31_i = f30_i + 1;
    f32_i = f31_i + 1;
    f33_i = f32_i + 1;
  }
}


template <class REAL>
static
void interpolate_bicubic_spline(REAL *pO, const REAL *pF, const REAL *pX, const REAL *pY, const mwSize ND, const mwSize M, const mwSize N, const mwSize P) {
  const mwSize LO = M*N;
  for (mwSize i=0; i<ND; ++i) {
    const REAL &x = pX[i];
    const REAL &y = pY[i];

    const REAL x_floor = floor(x) ;
    const REAL y_floor = floor(y) ;

    ///// Zach revised ///////////////
    const REAL dx = y-y_floor;
    const REAL dy = x-x_floor;
    ////////////////////////////

    const REAL dxx = dx*dx;
    const REAL dxxx = dxx*dx;

    const REAL dyy = dy*dy;
    const REAL dyyy = dyy*dy;

    const REAL wx0 = 0.5 * (    - dx + 2.0*dxx -       dxxx);
    const REAL wx1 = 0.5 * (2.0      - 5.0*dxx + 3.0 * dxxx);
    const REAL wx2 = 0.5 * (      dx + 4.0*dxx - 3.0 * dxxx);
    const REAL wx3 = 0.5 * (         -     dxx +       dxxx);


    const REAL wy0 = 0.5 * (    - dy + 2.0*dyy -       dyyy);
    const REAL wy1 = 0.5 * (2.0      - 5.0*dyy + 3.0 * dyyy);
    const REAL wy2 = 0.5 * (      dy + 4.0*dyy - 3.0 * dyyy);
    const REAL wy3 = 0.5 * (         -     dyy +       dyyy);

    int f00_i, f10_i, f20_i, f30_i, f01_i, f11_i, f21_i, f31_i; 
    int f02_i, f12_i, f22_i, f32_i, f03_i, f13_i, f23_i, f33_i;

    indices_cubic_spline(
        f00_i, f10_i, f20_i, f30_i, f01_i, f11_i, f21_i, f31_i, 
        f02_i, f12_i, f22_i, f32_i, f03_i, f13_i, f23_i, f33_i,
        int(x_floor-1), int(y_floor-1), M, N);

    for (mwSize j=0; j<P; ++j) {

        // [Q] matrix = [ pF[f00_i + j*LO], pF[f01_i + j*LO], pF[f02_i + j*LO], pF[f03_i + j*LO];
        //                pF[f10_i + j*LO], pF[f11_i + j*LO], pF[f12_i + j*LO], pF[f13_i + j*LO];
        //                pF[f20_i + j*LO], pF[f21_i + j*LO], pF[f22_i + j*LO], pF[f23_i + j*LO];
        //                pF[f30_i + j*LO], pF[f31_i + j*LO], pF[f32_i + j*LO], pF[f33_i + j*LO] ]
        
        REAL Q00 = pF[f00_i + j*LO];
        REAL Q01 = pF[f10_i + j*LO];
        REAL Q02 = pF[f20_i + j*LO];
        REAL Q03 = pF[f30_i + j*LO];
        REAL Q10 = pF[f01_i + j*LO];
        REAL Q11 = pF[f11_i + j*LO];
        REAL Q12 = pF[f21_i + j*LO];
        REAL Q13 = pF[f31_i + j*LO];
        REAL Q20 = pF[f02_i + j*LO];
        REAL Q21 = pF[f12_i + j*LO];
        REAL Q22 = pF[f22_i + j*LO];
        REAL Q23 = pF[f32_i + j*LO];
        REAL Q30 = pF[f03_i + j*LO];
        REAL Q31 = pF[f13_i + j*LO];
        REAL Q32 = pF[f23_i + j*LO];
        REAL Q33 = pF[f33_i + j*LO];

 
        // Previous codes for bicubic:
        // pO[i + j*ND] =
        //        wy0*(wx0 * pF[f00_i + j*LO] + wx1 * pF[f10_i + j*LO] +  wx2 * pF[f20_i + j*LO] + wx3 * pF[f30_i + j*LO]) +
        //        wy1*(wx0 * pF[f01_i + j*LO] + wx1 * pF[f11_i + j*LO] +  wx2 * pF[f21_i + j*LO] + wx3 * pF[f31_i + j*LO]) +
        //        wy2*(wx0 * pF[f02_i + j*LO] + wx1 * pF[f12_i + j*LO] +  wx2 * pF[f22_i + j*LO] + wx3 * pF[f32_i + j*LO]) +
        //        wy3*(wx0 * pF[f03_i + j*LO] + wx1 * pF[f13_i + j*LO] +  wx2 * pF[f23_i + j*LO] + wx3 * pF[f33_i + j*LO]);
      
      
      //////////////Zach revised////////////
        REAL a33 = 1.0f/49*(9*Q00 - 24*Q01 + 24*Q02 - 9*Q03 - 24*Q10 + 64*Q11 - 64*Q12 +  24*Q13 + 24*Q20 - 64*Q21 + 64*Q22 - 24*Q23 - 9*Q30 + 24*Q31 - 24*Q32 + 9*Q33);
        REAL a32 = -(3.0f/392)*(57*Q00 - 117*Q01 + 75*Q02 - 15*Q03 - 152*Q10 + 312*Q11 - 200*Q12 + 40*Q13 + 152*Q20 - 312*Q21 + 200*Q22 - 40*Q23 - 57*Q30 + 117*Q31 - 75*Q32 + 15*Q33);
        REAL a31 = 3.0f/392*(33*Q00 + 3*Q01 - 45*Q02 + 9*Q03 - 88*Q10 - 8*Q11 + 120*Q12 - 24*Q13 + 88*Q20 + 8*Q21 - 120*Q22 + 24*Q23 - 33*Q30 - 3*Q31 + 45*Q32 - 9*Q33);
        REAL a30 = 1.0f/7*(-3*Q01 + 8*Q11 - 8*Q21 + 3*Q31);
        REAL a23 = -(3.0f/392)*(57*Q00 - 152*Q01 + 152*Q02 - 57*Q03 - 117*Q10 + 312*Q11 - 312*Q12 + 117*Q13 + 75*Q20 - 200*Q21 + 200*Q22 - 75*Q23 - 15*Q30 + 40*Q31 - 40*Q32 + 15*Q33);
        REAL a22 = (1.0f/3136)*9*(361*Q00 - 741*Q01 + 475*Q02 - 95*Q03 - 741*Q10 + 1521*Q11 - 975*Q12 + 195*Q13 + 475*Q20 - 975*Q21 + 625*Q22 - 125*Q23 - 95*Q30 + 195*Q31 - 125*Q32 + 25*Q33);
        REAL a21 = -(1.0f/3136)*9*(209*Q00 + 19*Q01 - 285*Q02 + 57*Q03 - 429*Q10 - 39*Q11 + 585*Q12 - 117*Q13 + 275*Q20 + 25*Q21 - 375*Q22 + 75*Q23 - 55*Q30 - 5*Q31 + 75*Q32 - 15*Q33);
        REAL a20 = 3.0f/56*(19*Q01 - 39*Q11 + 25*Q21 - 5*Q31);
        REAL a13 = 3.0f/392*(33*Q00 - 88*Q01 + 88*Q02 - 33*Q03 + 3*Q10 - 8*Q11 + 8*Q12 - 3*Q13 - 45*Q20 + 120*Q21 - 120*Q22 + 45*Q23 + 9*Q30 - 24*Q31 + 24*Q32 - 9*Q33);
        REAL a12 = -(1.0f/3136)*9*(209*Q00 - 429*Q01 + 275*Q02 - 55*Q03 + 19*Q10 - 39*Q11 + 25*Q12 - 5*Q13 - 285*Q20 + 585*Q21 - 375*Q22 + 75*Q23 + 57*Q30 - 117*Q31 + 75*Q32 - 15*Q33);
        REAL a11= (1.0f/3136)*9*(121*Q00 + 11*Q01 - 165*Q02 + 33*Q03 + 11*Q10 + Q11 - 15*Q12 + 3*Q13 - 165*Q20 - 15*Q21 + 225*Q22 - 45*Q23 + 33*Q30 + 3*Q31 - 45*Q32 + 9*Q33);
        REAL a10 = -(3.0f/56)*(11*Q01 + Q11 + 3*(-5*Q21 + Q31));
        REAL a03 = 1.0f/7*(-3*Q10 + 8*Q11 - 8*Q12 + 3*Q13);
        REAL a02 = 3.0f/56*(19*Q10 - 39*Q11 + 25*Q12 - 5*Q13);
        REAL a01 = -(3.0f/56)*(11*Q10 + Q11 + 3*(-5*Q12 + Q13));
        REAL a00 = Q11;
        ////////////////////////////////////


        pO[i + j * ND] = a00 + a01 * dy + a02 * dyy + a03 * dyyy + a10 * dx + a11 * dx * dy + a12 * dx * dyy + a13 * dx * dyyy + a20 * dxx + a21 * dxx * dy + a22 * dxx * dyy + a23 * dxx * dyyy + a30 * dxxx + a31 * dxxx * dy + a32 * dxxx * dyy + a33 * dxxx * dyyy;


    }

  }
}

template <class REAL, size_t P>
static
void interpolate_bicubic_spline_unrolled(REAL *pO, const REAL *pF, const REAL *pX, const REAL *pY, const mwSize ND, mwSize M, mwSize N) {
  const mwSize LO = M*N;
  for (mwSize i=0; i<ND; ++i) {
    const REAL &x = pX[i];
    const REAL &y = pY[i];

    const REAL x_floor = floor(x) ;
    const REAL y_floor = floor(y) ;

    ///// Zach revised ///////////////
    const REAL dx = y-y_floor;
    const REAL dy = x-x_floor;
    ////////////////////////////

    const REAL dxx = dx*dx;
    const REAL dxxx = dxx*dx;

    const REAL dyy = dy*dy;
    const REAL dyyy = dyy*dy;

    //const REAL wx0 = 0.5 * (    - dx + 2.0*dxx -       dxxx);
    //const REAL wx1 = 0.5 * (2.0      - 5.0*dxx + 3.0 * dxxx);
    //const REAL wx2 = 0.5 * (      dx + 4.0*dxx - 3.0 * dxxx);
    //const REAL wx3 = 0.5 * (         -     dxx +       dxxx);


    //const REAL wy0 = 0.5 * (    - dy + 2.0*dyy -       dyyy);
    //const REAL wy1 = 0.5 * (2.0      - 5.0*dyy + 3.0 * dyyy);
    //const REAL wy2 = 0.5 * (      dy + 4.0*dyy - 3.0 * dyyy);
    //const REAL wy3 = 0.5 * (         -     dyy +       dyyy);


    int f00_i, f10_i, f20_i, f30_i, f01_i, f11_i, f21_i, f31_i; 
    int f02_i, f12_i, f22_i, f32_i, f03_i, f13_i, f23_i, f33_i;

    indices_cubic_spline(
        f00_i, f10_i, f20_i, f30_i, f01_i, f11_i, f21_i, f31_i, 
        f02_i, f12_i, f22_i, f32_i, f03_i, f13_i, f23_i, f33_i,
        int(x_floor-1), int(y_floor-1), M, N);

    for (mwSize j=0; j<P; ++j) {

        // [Q] matrix = [ pF[f00_i + j*LO], pF[f01_i + j*LO], pF[f02_i + j*LO], pF[f03_i + j*LO];
        //                pF[f10_i + j*LO], pF[f11_i + j*LO], pF[f12_i + j*LO], pF[f13_i + j*LO];
        //                pF[f20_i + j*LO], pF[f21_i + j*LO], pF[f22_i + j*LO], pF[f23_i + j*LO];
        //                pF[f30_i + j*LO], pF[f31_i + j*LO], pF[f32_i + j*LO], pF[f33_i + j*LO] ]
        
        REAL Q00 = pF[f00_i + j*LO];
        REAL Q01 = pF[f10_i + j*LO];
        REAL Q02 = pF[f20_i + j*LO];
        REAL Q03 = pF[f30_i + j*LO];
        REAL Q10 = pF[f01_i + j*LO];
        REAL Q11 = pF[f11_i + j*LO];
        REAL Q12 = pF[f21_i + j*LO];
        REAL Q13 = pF[f31_i + j*LO];
        REAL Q20 = pF[f02_i + j*LO];
        REAL Q21 = pF[f12_i + j*LO];
        REAL Q22 = pF[f22_i + j*LO];
        REAL Q23 = pF[f32_i + j*LO];
        REAL Q30 = pF[f03_i + j*LO];
        REAL Q31 = pF[f13_i + j*LO];
        REAL Q32 = pF[f23_i + j*LO];
        REAL Q33 = pF[f33_i + j*LO];

 
        // Previous codes for bicubic:
        // pO[i + j*ND] =
        //        wy0*(wx0 * pF[f00_i + j*LO] + wx1 * pF[f10_i + j*LO] +  wx2 * pF[f20_i + j*LO] + wx3 * pF[f30_i + j*LO]) +
        //        wy1*(wx0 * pF[f01_i + j*LO] + wx1 * pF[f11_i + j*LO] +  wx2 * pF[f21_i + j*LO] + wx3 * pF[f31_i + j*LO]) +
        //        wy2*(wx0 * pF[f02_i + j*LO] + wx1 * pF[f12_i + j*LO] +  wx2 * pF[f22_i + j*LO] + wx3 * pF[f32_i + j*LO]) +
        //        wy3*(wx0 * pF[f03_i + j*LO] + wx1 * pF[f13_i + j*LO] +  wx2 * pF[f23_i + j*LO] + wx3 * pF[f33_i + j*LO]);
       
        //////////////Zach revised////////////
        REAL a33 = 1.0f/49*(9*Q00 - 24*Q01 + 24*Q02 - 9*Q03 - 24*Q10 + 64*Q11 - 64*Q12 +  24*Q13 + 24*Q20 - 64*Q21 + 64*Q22 - 24*Q23 - 9*Q30 + 24*Q31 - 24*Q32 + 9*Q33);
        REAL a32 = -(3.0f/392)*(57*Q00 - 117*Q01 + 75*Q02 - 15*Q03 - 152*Q10 + 312*Q11 - 200*Q12 + 40*Q13 + 152*Q20 - 312*Q21 + 200*Q22 - 40*Q23 - 57*Q30 + 117*Q31 - 75*Q32 + 15*Q33);
        REAL a31 = 3.0f/392*(33*Q00 + 3*Q01 - 45*Q02 + 9*Q03 - 88*Q10 - 8*Q11 + 120*Q12 - 24*Q13 + 88*Q20 + 8*Q21 - 120*Q22 + 24*Q23 - 33*Q30 - 3*Q31 + 45*Q32 - 9*Q33);
        REAL a30 = 1.0f/7*(-3*Q01 + 8*Q11 - 8*Q21 + 3*Q31);
        REAL a23 = -(3.0f/392)*(57*Q00 - 152*Q01 + 152*Q02 - 57*Q03 - 117*Q10 + 312*Q11 - 312*Q12 + 117*Q13 + 75*Q20 - 200*Q21 + 200*Q22 - 75*Q23 - 15*Q30 + 40*Q31 - 40*Q32 + 15*Q33);
        REAL a22 = (1.0f/3136)*9*(361*Q00 - 741*Q01 + 475*Q02 - 95*Q03 - 741*Q10 + 1521*Q11 - 975*Q12 + 195*Q13 + 475*Q20 - 975*Q21 + 625*Q22 - 125*Q23 - 95*Q30 + 195*Q31 - 125*Q32 + 25*Q33);
        REAL a21 = -(1.0f/3136)*9*(209*Q00 + 19*Q01 - 285*Q02 + 57*Q03 - 429*Q10 - 39*Q11 + 585*Q12 - 117*Q13 + 275*Q20 + 25*Q21 - 375*Q22 + 75*Q23 - 55*Q30 - 5*Q31 + 75*Q32 - 15*Q33);
        REAL a20 = 3.0f/56*(19*Q01 - 39*Q11 + 25*Q21 - 5*Q31);
        REAL a13 = 3.0f/392*(33*Q00 - 88*Q01 + 88*Q02 - 33*Q03 + 3*Q10 - 8*Q11 + 8*Q12 - 3*Q13 - 45*Q20 + 120*Q21 - 120*Q22 + 45*Q23 + 9*Q30 - 24*Q31 + 24*Q32 - 9*Q33);
        REAL a12 = -(1.0f/3136)*9*(209*Q00 - 429*Q01 + 275*Q02 - 55*Q03 + 19*Q10 - 39*Q11 + 25*Q12 - 5*Q13 - 285*Q20 + 585*Q21 - 375*Q22 + 75*Q23 + 57*Q30 - 117*Q31 + 75*Q32 - 15*Q33);
        REAL a11= (1.0f/3136)*9*(121*Q00 + 11*Q01 - 165*Q02 + 33*Q03 + 11*Q10 + Q11 - 15*Q12 + 3*Q13 - 165*Q20 - 15*Q21 + 225*Q22 - 45*Q23 + 33*Q30 + 3*Q31 - 45*Q32 + 9*Q33);
        REAL a10 = -(3.0f/56)*(11*Q01 + Q11 + 3*(-5*Q21 + Q31));
        REAL a03 = 1.0f/7*(-3*Q10 + 8*Q11 - 8*Q12 + 3*Q13);
        REAL a02 = 3.0f/56*(19*Q10 - 39*Q11 + 25*Q12 - 5*Q13);
        REAL a01 = -(3.0f/56)*(11*Q10 + Q11 + 3*(-5*Q12 + Q13));
        REAL a00 = Q11;
        ////////////////////////////////////

       pO[i + j*ND] = a00 + a01*dy + a02*dyy + a03*dyyy + a10*dx + a11*dx*dy + a12*dx*dyy + a13*dx*dyyy + a20*dxx + a21*dxx*dy + a22*dxx*dyy + a23*dxx*dyyy + a30*dxxx + a31*dxxx*dy + a32*dxxx*dyy + a33*dxxx*dyyy;


    }

  }
}

enum InterpolationMethod { cubicspline };

// static
// InterpolationMethod parseInterpolationMethod(int nrhs, const mxArray *prhs[]) {
//   if (nrhs<4)
//     return Cubic;
// 
//   char method[10] = "cubic    ";
// 
//   mxGetString(prhs[3], method, 9);
// 
//   if (std::string(method).substr(0, 7).compare("nearest")==0)
//     return Nearest;
//   else if (std::string(method).substr(0, 6).compare("linear")==0)
//     return Linear;
//   else if (std::string(method).substr(0, 5).compare("cubic")==0)
//     return Cubic;
//   else
//     mexErrMsgTxt("Specify one of nearest, linear, cubic as the interpolation method argument.");
// 
//   return(Cubic);
// }

template <class REAL> 
static void interpolate(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  const mwSize *F_dims = mxGetDimensions(prhs[0]);
  const mwSize *X_dims = mxGetDimensions(prhs[1]);
  const mwSize *Y_dims = mxGetDimensions(prhs[2]);

  if (mxGetNumberOfDimensions(prhs[1]) != mxGetNumberOfDimensions(prhs[2]))
    mexErrMsgTxt("X, Y should have the same size");

  const mwSize M=F_dims[0];
  const mwSize N=F_dims[1];
  mwSize P=1;

  mwSize outDims[50];
  if (mxGetNumberOfDimensions(prhs[2]) + mxGetNumberOfDimensions(prhs[0]) - 2 > 50)
    mexErrMsgTxt("Can't have that many dimensions in interpolated data.");

  for (mwSize i=0; i<mxGetNumberOfDimensions(prhs[1]); ++i) {
    if (X_dims[i] != Y_dims[i])
      mexErrMsgTxt("X, Y should have the same size");
    outDims[i] = X_dims[i];
  }
  for (mwSize i=2; i<mxGetNumberOfDimensions(prhs[0]); ++i) {
    outDims[mxGetNumberOfDimensions(prhs[1])+i-2] = F_dims[i];
    P *= F_dims[i];
  }

  plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[2]) + mxGetNumberOfDimensions(prhs[0]) - 2, outDims, mxIsSingle(prhs[0]) ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);

  const mwSize ND = mxGetNumberOfElements(prhs[1]);

  const REAL *pF = (REAL*)mxGetData(prhs[0]);
  const REAL *pX = (REAL*)mxGetData(prhs[1]);
  const REAL *pY = (REAL*)mxGetData(prhs[2]);
  REAL       *pO = (REAL*)mxGetData(plhs[0]);  

 // switch(parseInterpolationMethod(nrhs, prhs)) {
    
 //   case CubicSpline:
      switch (P) {
        case 1: interpolate_bicubic_spline_unrolled<REAL, 1>(pO, pF, pX, pY, ND, M, N); break;
        case 2: interpolate_bicubic_spline_unrolled<REAL, 2>(pO, pF, pX, pY, ND, M, N); break;
        case 3: interpolate_bicubic_spline_unrolled<REAL, 3>(pO, pF, pX, pY, ND, M, N); break;
        case 4: interpolate_bicubic_spline_unrolled<REAL, 4>(pO, pF, pX, pY, ND, M, N); break;
        case 5: interpolate_bicubic_spline_unrolled<REAL, 5>(pO, pF, pX, pY, ND, M, N); break;
        case 6: interpolate_bicubic_spline_unrolled<REAL, 6>(pO, pF, pX, pY, ND, M, N); break;
        case 7: interpolate_bicubic_spline_unrolled<REAL, 7>(pO, pF, pX, pY, ND, M, N); break;
        case 8: interpolate_bicubic_spline_unrolled<REAL, 8>(pO, pF, pX, pY, ND, M, N); break;
        case 9: interpolate_bicubic_spline_unrolled<REAL, 9>(pO, pF, pX, pY, ND, M, N); break;
        default: 
                interpolate_bicubic_spline<REAL>(pO, pF, pX, pY, ND, M, N, P);
      }
  // break;
  // default:
  // mexErrMsgTxt("Unimplemented interpolation method.");
  // }
}

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {

  if((nrhs!=3) && (nrhs!=4))
    mexErrMsgTxt("Wrong number of input arguments for Z = ba_interp2(F, X, Y, [method])");

  if(nlhs>1)
    mexErrMsgTxt("Wrong number of output arguments for Z = ba_interp2(F, X, Y, [method])");

  if (mxIsDouble(prhs[0]) && mxIsDouble(prhs[1]) && mxIsDouble(prhs[2]))
    // All is double, all is well
    interpolate<double>(nlhs, plhs, nrhs, prhs);
  else if (mxIsSingle(prhs[0]) && mxIsSingle(prhs[1]) && mxIsSingle(prhs[2]))
    // All is single, all is well
    interpolate<float>(nlhs, plhs, nrhs, prhs);
  else
    mexErrMsgTxt("ba_interp2 takes only double or single arguments for IMG,X,Y, and all must have the same class");
}
