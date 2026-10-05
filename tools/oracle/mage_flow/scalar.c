// Independent C scalar primitives. No BLAS, contraction, vectorization or approximate math.
#include "../laya/scalar.c"

void mage_unary(const float*x,float*y,size_t n,int op){
    for(size_t i=0;i<n;i++){
        float v=x[i];
        switch(op){
        case 0:y[i]=1.0f/sqrtf(v);break;
        case 1:y[i]=v/(1.0f+expf(-v));break;
        case 2:y[i]=1.0f/(1.0f+expf(-v));break;
        case 3:y[i]=v*v;break;
        case 4:y[i]=0.5f*v*(1.0f+tanhf(sqrtf(2.0f/3.14159265358979323846f)*(v+0.044715f*v*v*v)));break;
        case 5:y[i]=sqrtf(v);break;
        default:abort();
        }
    }
}
void mage_mean(const float*x,float*y,size_t rows,size_t d){
    for(size_t r=0;r<rows;r++){double sum=0;for(size_t i=0;i<d;i++)sum+=(double)x[r*d+i];y[r]=(float)(sum/(double)d);}
}
void mage_norm(const float*x,const float*w,const float*b,float*y,float*means,float*scales,size_t rows,size_t d,float eps){
    for(size_t r=0;r<rows;r++){
        double sum=0;for(size_t i=0;i<d;i++)sum+=(double)x[r*d+i];float mean=(float)(sum/(double)d);
        double var=0;for(size_t i=0;i<d;i++){float delta=x[r*d+i]-mean;var+=(double)(delta*delta);}
        float scale=1.0f/sqrtf((float)(var/(double)d)+eps);
        if(means)means[r]=mean;if(scales)scales[r]=scale;
        for(size_t i=0;i<d;i++)y[r*d+i]=(x[r*d+i]-mean)*scale*(w?w[i]:1.0f)+(b?b[i]:0.0f);
    }
}
void mage_group_norm(const float*x,const float*w,const float*b,float*y,size_t batches,size_t c,size_t plane,size_t groups,float eps){
    size_t cg=c/groups,d=cg*plane;
    for(size_t batch=0;batch<batches;batch++)for(size_t g=0;g<groups;g++){
        const float* row=x+batch*c*plane+g*d;double sum=0;for(size_t i=0;i<d;i++)sum+=(double)row[i];float mean=(float)(sum/(double)d);
        double var=0;for(size_t i=0;i<d;i++){float delta=row[i]-mean;var+=(double)(delta*delta);}
        float scale=1.0f/sqrtf((float)(var/(double)d)+eps);
        for(size_t ch=0;ch<cg;ch++)for(size_t i=0;i<plane;i++){size_t at=batch*c*plane+(g*cg+ch)*plane+i;y[at]=(x[at]-mean)*scale*w[g*cg+ch]+b[g*cg+ch];}
    }
}
void mage_conv(const float*x,const float*w,const float*b,float*y,size_t batches,size_t ci,size_t h,size_t width,size_t co,size_t kh,size_t kw,size_t sh,size_t sw,size_t ph,size_t pw,size_t groups){
    size_t oh=(h+2*ph-kh)/sh+1,ow=(width+2*pw-kw)/sw+1,icg=ci/groups,ocg=co/groups;
    for(size_t batch=0;batch<batches;batch++)for(size_t oc=0;oc<co;oc++)for(size_t yy=0;yy<oh;yy++)for(size_t xx=0;xx<ow;xx++){
        float sum=b?b[oc]:0.0f;
        for(size_t ch=0;ch<icg;ch++)for(size_t ky=0;ky<kh;ky++){
            long iy=(long)(yy*sh+ky)-(long)ph;if(iy<0||iy>=(long)h)continue;
            // The shared scalar dot rounds each product to F32, then sums in F64.
            double dot=0;
            for(size_t kx=0;kx<kw;kx++){long ix=(long)(xx*sw+kx)-(long)pw;if(ix>=0&&ix<(long)width)dot+=(double)(x[((batch*ci+(oc/ocg)*icg+ch)*h+iy)*width+ix]*w[((oc*icg+ch)*kh+ky)*kw+kx]);}
            sum+=(float)dot;
        }
        y[((batch*co+oc)*oh+yy)*ow+xx]=sum;
    }
}
void mage_complex_mul(const float*x,const float*z,float*y,size_t n){
    for(size_t i=0;i<n;i++){float a=x[2*i],b=x[2*i+1],c=z[2*i],d=z[2*i+1];y[2*i]=a*c-b*d;y[2*i+1]=a*d+b*c;}
}

void mage_linspace(float start,float end,float*y,size_t n){
    float step=(end-start)/(float)(n-1);
    for(size_t i=0;i<n;i++)y[i]=n==1?start:(i<n/2?start+step*(float)i:end-step*(float)(n-1-i));
}

void mage_linear64(const float*x,const float*w,const float*b,float*y,size_t rows,size_t in,size_t out){
 for(size_t r=0;r<rows;r++)for(size_t o=0;o<out;o++){double v=0;for(size_t k=0;k<in;k++)v+=(double)(x[r*in+k]*w[o*in+k]);y[r*out+o]=(float)v+(b?b[o]:0.0f);}
}
void mage_softmax64(const float*x,float*y,size_t rows,size_t d){
 for(size_t r=0;r<rows;r++){float max=-INFINITY;double sum=0;for(size_t i=0;i<d;i++)max=fmaxf(max,x[r*d+i]);for(size_t i=0;i<d;i++){y[r*d+i]=expf(x[r*d+i]-max);sum+=(double)y[r*d+i];}float scale=(float)(1.0/sum);for(size_t i=0;i<d;i++)y[r*d+i]*=scale;}
}
void mage_attention(const float*q,const float*k,const float*v,float*y,size_t batches,size_t nq,size_t nk,size_t d,float scale,int score64,int softmax64,int causal){
 float*scores=malloc(nk*sizeof(float));if(!scores)abort();
 for(size_t b=0;b<batches;b++)for(size_t i=0;i<nq;i++){
  for(size_t j=0;j<nk;j++){
   if(causal&&j>i){scores[j]=-INFINITY;continue;}
   float sum=0;double wide=0;for(size_t a=0;a<d;a++){float product=q[(b*nq+i)*d+a]*k[(b*nk+j)*d+a];if(score64)wide+=(double)product;else sum+=product;}
   scores[j]=(score64?(float)wide:sum)*scale;
  }
  if(softmax64)mage_softmax64(scores,scores,1,nk);else softmax(scores,scores,1,nk);
  for(size_t a=0;a<d;a++){float sum=0;for(size_t j=0;j<nk;j++)sum+=scores[j]*v[(b*nk+j)*d+a];y[(b*nq+i)*d+a]=sum;}
 }
 free(scores);
}
void mage_patch(const float*x,const float*w,const float*b,float*y,size_t rows,size_t d,size_t out){mage_linear64(x,w,b,y,rows,d,out);}
