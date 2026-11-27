// Independent scalar arithmetic for the pinned official Astronex graph.
// No contraction, vectorization, reassociation, BLAS, or approximate exp/tanh.
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#define PRIMITIVES(T, S, EXP, LOG, SIN, COS, TANH, SQRT, POW, ERF, ABS, FLOOR, EXPM1) \
void linear_##S(const T *x, const T *w, const T *b, T *y, size_t rows, size_t in, size_t out) { \
    for (size_t r=0;r<rows;r++) for (size_t o=0;o<out;o++) { \
        T sum=0; for (size_t k=0;k<in;k++) sum += x[r*in+k]*w[o*in+k]; \
        y[r*out+o] = b ? sum+b[o] : sum; \
    } \
} \
void unary_##S(const T *x, T *y, size_t n, int op, T a) { \
    for (size_t i=0;i<n;i++) { T v=x[i]; switch(op) { \
        case 0:y[i]=-v;break; case 1:y[i]=EXP(v);break; case 2:y[i]=LOG(v);break; \
        case 3:y[i]=SIN(v);break; case 4:y[i]=COS(v);break; case 5:y[i]=TANH(v);break; \
        case 6:y[i]=SQRT(v);break; case 7:y[i]=(T)1/SQRT(v);break; \
        case 8:y[i]=POW(v,a);break; case 9:y[i]=POW(a,v);break; \
        case 10:y[i]=(T)1/v;break; case 11:y[i]=ABS(v);break; case 12:y[i]=FLOOR(v);break; \
        case 13:y[i]=EXPM1(v);break; case 14:y[i]=(T)1/((T)1+EXP(-v));break; \
        case 15:y[i]=v/((T)1+EXP(-v));break; \
        case 16:y[i]=((T)0.5*v)*((T)1+ERF(v*(T)0.7071067811865475));break; \
        case 17:{ T cube=(v*v)*v; T z=(T)0.7978845608028654*(v+(T)0.044715*cube); \
                  y[i]=((T)0.5*v)*((T)1+TANH(z));break;} \
        default:abort(); \
    }} \
} \
void binary_##S(const T *x, const T *z, T *y, size_t n, int op) { \
    for (size_t i=0;i<n;i++) switch(op) { \
        case 0:y[i]=x[i]+z[i];break; case 1:y[i]=x[i]-z[i];break; \
        case 2:y[i]=x[i]*z[i];break; case 3:y[i]=x[i]/z[i];break; \
        case 4:y[i]=x[i]>z[i]?x[i]:z[i];break; case 5:y[i]=x[i]<z[i]?x[i]:z[i];break; \
        case 6:y[i]=POW(x[i],z[i]);break; default:abort(); \
    } \
} \
void reduce_##S(const T *x, T *y, size_t rows, size_t d, int op) { \
    for (size_t r=0;r<rows;r++) { T sum=op==2?-INFINITY:op==3?INFINITY:0; \
        for (size_t i=0;i<d;i++) { T v=x[r*d+i]; \
            if(op==2) {if(v>sum)sum=v;} else if(op==3) {if(v<sum)sum=v;} \
            else sum += op==4 ? v*v : v; \
        } y[r]=op==1?sum/(T)d:op==4?SQRT(sum):sum; \
    } \
} \
void norm_##S(const T *x, const T *w, const T *b, T *y, T *means, T *invstd, size_t rows, size_t d, T eps) { \
    for (size_t r=0;r<rows;r++) { T mean=0,var=0; \
        for(size_t i=0;i<d;i++)mean+=x[r*d+i]; mean/=(T)d; \
        for(size_t i=0;i<d;i++){T delta=x[r*d+i]-mean;var+=delta*delta;} var/=(T)d; \
        T scale=(T)1/SQRT(var+eps); if(means)means[r]=mean;if(invstd)invstd[r]=scale; \
        for(size_t i=0;i<d;i++){T v=(x[r*d+i]-mean)*scale; \
            if(w)v*=w[i];if(b)v+=b[i];y[r*d+i]=v;} \
    } \
} \
void softmax_##S(const T *x, T *y, size_t rows, size_t d) { \
    for(size_t r=0;r<rows;r++){T max=-INFINITY,sum=0; \
        for(size_t i=0;i<d;i++)if(x[r*d+i]>max)max=x[r*d+i]; \
        for(size_t i=0;i<d;i++){y[r*d+i]=EXP(x[r*d+i]-max);sum+=y[r*d+i];} \
        for(size_t i=0;i<d;i++)y[r*d+i]/=sum; \
    } \
}

PRIMITIVES(float, f32, expf, logf, sinf, cosf, tanhf, sqrtf, powf, erff, fabsf, floorf, expm1f)
PRIMITIVES(double, f64, exp, log, sin, cos, tanh, sqrt, pow, erf, fabs, floor, expm1)

void complex_mul(const double *x, const double *z, double *y, size_t n) {
    for(size_t i=0;i<n;i++) { double a=x[2*i],b=x[2*i+1],c=z[2*i],d=z[2*i+1];
        y[2*i]=a*c-b*d;y[2*i+1]=a*d+b*c;
    }
}

void polar(const double *radius, const double *angle, double *y, size_t n) {
    for(size_t i=0;i<n;i++){y[2*i]=radius[i]*cos(angle[i]);y[2*i+1]=radius[i]*sin(angle[i]);}
}

void attention(const float *q,const float *k,const float *v,const float *mask,float *y,
               size_t batches,size_t nq,size_t nk,size_t d,size_t dv,float scale) {
    float *scores=malloc(nk*sizeof(float));if(!scores)abort();
    for(size_t b=0;b<batches;b++)for(size_t i=0;i<nq;i++){
        for(size_t j=0;j<nk;j++){float sum=0;
            for(size_t a=0;a<d;a++)sum+=q[(b*nq+i)*d+a]*k[(b*nk+j)*d+a];
            scores[j]=sum*scale;if(mask)scores[j]+=mask[(b*nq+i)*nk+j];
        }
        softmax_f32(scores,scores,1,nk);
        for(size_t a=0;a<dv;a++){float sum=0;
            for(size_t j=0;j<nk;j++)sum+=scores[j]*v[(b*nk+j)*dv+a];y[(b*nq+i)*dv+a]=sum;
        }
    }free(scores);
}

void project_f32(const float *matrix, const float *x, float *y,
                 size_t batch, size_t heads, size_t cameras,
                 size_t patches, size_t groups, size_t dim) {
    for (size_t b=0;b<batch;b++) for (size_t n=0;n<heads;n++)
    for (size_t c=0;c<cameras;c++) for (size_t p=0;p<patches;p++)
    for (size_t g=0;g<groups;g++) for (size_t i=0;i<dim;i++) {
        float sum=0;
        for (size_t j=0;j<dim;j++) {
            size_t mi=(((b*cameras+c)*dim+i)*dim+j);
            size_t xi=((((((b*heads+n)*cameras+c)*patches+p)*groups+g)*dim)+j);
            sum += matrix[mi] * x[xi];
        }
        size_t yi=((((((b*heads+n)*cameras+c)*patches+p)*groups+g)*dim)+i);
        y[yi]=sum;
    }
}

void conv3d(const float *x,const float *w,const float *b,float *y,const size_t *s) {
    size_t B=s[0],IC=s[1],ID=s[2],IH=s[3],IW=s[4],OC=s[5],KD=s[6],KH=s[7],KW=s[8];
    size_t OD=s[9],OH=s[10],OW=s[11],SD=s[12],SH=s[13],SW=s[14],PD=s[15],PH=s[16],PW=s[17];
    size_t DD=s[18],DH=s[19],DW=s[20],G=s[21],IG=IC/G,OG=OC/G;
    for(size_t batch=0;batch<B;batch++)for(size_t oc=0;oc<OC;oc++)
    for(size_t od=0;od<OD;od++)for(size_t oh=0;oh<OH;oh++)for(size_t ow=0;ow<OW;ow++){
        float sum=0;for(size_t ic=0;ic<IG;ic++)for(size_t kd=0;kd<KD;kd++)
        for(size_t kh=0;kh<KH;kh++)for(size_t kw=0;kw<KW;kw++){
            ptrdiff_t iz=od*SD+kd*DD-PD,iy=oh*SH+kh*DH-PH,ix=ow*SW+kw*DW-PW;
            if(iz<0||iy<0||ix<0||(size_t)iz>=ID||(size_t)iy>=IH||(size_t)ix>=IW)continue;
            size_t c=oc/OG*IG+ic;
            sum+=x[(((batch*IC+c)*ID+iz)*IH+iy)*IW+ix]*w[(((oc*IG+ic)*KD+kd)*KH+kh)*KW+kw];
        }y[(((batch*OC+oc)*OD+od)*OH+oh)*OW+ow]=b?sum+b[oc]:sum;
    }
}

// Only small UniPC coefficient systems use this solver, never model matmuls.
void solve_f32(const float *a,const float *b,float *y,size_t n) {
    float *m=malloc(n*(n+1)*sizeof(float));if(!m)abort();
    for(size_t i=0;i<n;i++){for(size_t j=0;j<n;j++)m[i*(n+1)+j]=a[i*n+j];m[i*(n+1)+n]=b[i];}
    for(size_t k=0;k<n;k++){
        size_t pivot=k;for(size_t i=k+1;i<n;i++)if(fabsf(m[i*(n+1)+k])>fabsf(m[pivot*(n+1)+k]))pivot=i;
        if(pivot!=k)for(size_t j=k;j<=n;j++){float tmp=m[k*(n+1)+j];m[k*(n+1)+j]=m[pivot*(n+1)+j];m[pivot*(n+1)+j]=tmp;}
        for(size_t i=k+1;i<n;i++){float scale=m[i*(n+1)+k]/m[k*(n+1)+k];
            for(size_t j=k;j<=n;j++)m[i*(n+1)+j]-=scale*m[k*(n+1)+j];}
    }
    for(size_t ii=n;ii>0;ii--){size_t i=ii-1;float value=m[i*(n+1)+n];
        for(size_t j=i+1;j<n;j++)value-=m[i*(n+1)+j]*y[j];y[i]=value/m[i*(n+1)+i];}free(m);
}
