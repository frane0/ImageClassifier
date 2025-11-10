#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define IMAGE_SIZE 28
#define VECTOR_SIZE 785
#define DATASET_SIZE 800
#define TESTSET_SIZE 200
#define STEP_SIZE 0.0001
#define MINI_BATCH_SIZE 80
#define MAX_ITERATION 1000


void load_pgm(const char *filename, unsigned char **image) //ok
{
    FILE *file = fopen(filename, "r");
    char header[3];
    int width, height, max_value;

    if (!file)
    {
        perror("File");
        exit(EXIT_FAILURE);
    }

    // PGM basligini oku (P2 formati olmali)
    fscanf(file, "%2s", header);
    fscanf(file, "%d %d", &width, &height);
    fscanf(file, "%d", &max_value);

    if (width != IMAGE_SIZE || height != IMAGE_SIZE)
    {
        printf("Wrong pixel size: %d x %d\n", width, height);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Piksel degerlerini oku
    for (int i = 0; i < IMAGE_SIZE; i++)
    {
        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            fscanf(file, "%hhu", &image[i][j]);
        }
    }

    fclose(file);
}

void print_image(unsigned char **image)
{
    for (int i = 0; i < IMAGE_SIZE; i++)
    {
        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            printf("%3d ", image[i][j]);
        }
        printf("\n");
    }
}

char *imageName(int i, char trainOrTest)
{
    char *fileName=(char *)calloc(8,sizeof(char));
    switch (i){
    case 0 ... 9:
        fileName[0]=i+48; //integer olan i'yi char'a cevir
        fileName[1]=0;
        break;
    case 10 ... 99:
        fileName[0]=i/10+48;
        fileName[1]=i%10+48;
        fileName[2]=0;
        break;
    default:
        fileName[0]=i/100+48;
        fileName[1]=(i%100)/10+48;
        fileName[2]=(i%100)%10+48;
        fileName[3]=0;
        break;
    }
    if(trainOrTest==1)
    {
        fileName=strcat(fileName,"_train.pgm");
    }
    else{
        fileName=strcat(fileName,"_test.pgm");
    }

    return fileName;
}

double vectorProduct(double *first, double *second, int vectorLength)
{
    double result=0;
    for(int i=0;i<vectorLength;i++)
    {
        result+=first[i]*second[i];
    }
    return result;
}

void calculateResidual(double *residual, double *W, double **X, double *Y)
{
    for(int i=0;i<DATASET_SIZE;i++)
    {
        residual[i]=Y[i]-tanh(vectorProduct(W,X[i],VECTOR_SIZE));
    }
}

void calculateJacobian(double **jacobian, double *W, double **X) //ok
{
    double productResult;
    double a,x;
    for(int i=0;i<DATASET_SIZE;i++)
    {
        productResult=0;
        for(int j=0;j<VECTOR_SIZE;j++)
        {
            productResult+=W[j]*X[i][j];
        }

        for(int j=0;j<VECTOR_SIZE;j++)
        {
            jacobian[i][j]= (-1)*X[i][j] * pow( 1/cosh(productResult) , 2);
        }
    }
}

int test(double *W, double **X, double *Y)
{
    double estimated;
    int result=0;
    for(int i=0;i<TESTSET_SIZE;i++)
    {
        estimated=tanh(vectorProduct(W,X[i],VECTOR_SIZE));
        printf("TEST %d: Real: %lf  /  Estimated: %lf\n",i,Y[i],estimated);
        if(Y[i]*estimated>=0)
            result++;
    }
    return result;
}

double testError(double *W, double **X, double *Y)
{
    double estimated;
    double result=0;
    for(int i=0;i<TESTSET_SIZE;i++)
    {
        estimated=tanh(vectorProduct(W,X[i],VECTOR_SIZE));
        //printf("TEST %d: Real: %lf  /  Estimated: %lf\n",i,Y[i],estimated);
        result+=abs(estimated-Y[i]);
    }
    result/=TESTSET_SIZE;
    return result;
}

void gradientDescent(double *W, double **X, double *Y, double eps, double **errorTime, double **allWValues, double **X_test, double *Y_test)
{
    int     iter=0;
    double  error=1;
    double  error_test=0;
    double  totalDiff=10;
    struct  timespec start, now;

    double *residual=(double *)calloc(DATASET_SIZE,sizeof(double));
    double *temp    =(double *)calloc(VECTOR_SIZE,sizeof(double));

    double **jacobian=(double **)calloc(DATASET_SIZE,sizeof(double *));
    for(int i=0;i<DATASET_SIZE;i++)
        jacobian[i]=(double *)calloc(VECTOR_SIZE,sizeof(double));

    calculateResidual(residual,W,X,Y);
    clock_gettime(CLOCK_MONOTONIC, &start);

    while(totalDiff>0.000001 && error>0.05 && iter<MAX_ITERATION)
    {
            //W DEGERLERINI KAYDET

        for(int i=0;i<VECTOR_SIZE;i++)
        {
            allWValues[iter][i]=W[i];
        }
        totalDiff=0;
        printf("\nITERATION %d\n",iter+1);

            //YENI W HESAPLANMASI

        calculateJacobian(jacobian,W,X);
        for(int i=0;i<VECTOR_SIZE;i++)
        {
            temp[i]=0;
            for(int j=0;j<DATASET_SIZE;j++)
                temp[i]+=jacobian[j][i]*residual[j];

            temp[i]=temp[i]*2*eps;
        }
        for(int i=0;i<VECTOR_SIZE;i++)
        {
            W[i]=W[i]-temp[i];
            //printf("New W%d=%lf\n",i,W[i]);
            totalDiff+=fabs(temp[i]);
        }

            //ERROR HESAPLANMASI

        calculateResidual(residual,W,X,Y);
        error=0;
        for(int i=0;i<DATASET_SIZE;i++)
            error+=pow(residual[i],2);
        error/=DATASET_SIZE;
        totalDiff/=VECTOR_SIZE;
        error_test=testError(W,X_test,Y_test);
        printf("\nTrain Error: %lf\nTest Error: %lf\nCurrent Difference: %lf\n---------------------\n---------------------\n---------------------\n",error,error_test,totalDiff);

            //ERRORLARI KAYDET

        errorTime[iter][0]=error;
        errorTime[iter][1]=error_test;
        clock_gettime(CLOCK_MONOTONIC, &now);
        errorTime[iter][2]=(now.tv_sec - start.tv_sec) * 1000.0+(now.tv_nsec - start.tv_nsec) / 1e6;
        printf("TIME: %lf // ERROR: %lf\n",errorTime[iter][1],errorTime[iter][0]);
        iter++;
    }
    if(error<=0.05)
        printf("Function converged with %d iterations! Last error rate:%lf\n",iter,error);
    else
        printf("Function did not converge! Last error rate:%lf",error);
    errorTime[iter]=NULL;
    for(int i=0;i<VECTOR_SIZE;i++)
    {
        allWValues[iter][i]=W[i];
    }
    allWValues[iter+1]=NULL;
}



void calculateResidual_SGD(double *residual, double *W, double **X, double *Y)
{
    for(int i=0;i<MINI_BATCH_SIZE;i++)
        residual[i]=Y[i]-tanh(vectorProduct(W,X[i],VECTOR_SIZE));
}

void calculateJacobian_SGD(double **jacobian, double *W, double **X)
{
    double productResult;
    double a,x;
    for(int i=0;i<MINI_BATCH_SIZE;i++)
    {
        productResult=0;
        for(int k=0;k<VECTOR_SIZE;k++)
        {
            productResult+=W[k]*X[i][k];

        }
        for(int j=0;j<VECTOR_SIZE;j++)
        {

            a=X[i][j];
            jacobian[i][j]= (-1)*a * pow( 1/(cosh(productResult)) , 2);
        }
    }
}

void stochasticGradientDescent(double *W, double **X_Full, double *Y_Full, double eps, double **errorTime, double **allWValues, double **X_test, double *Y_test)
{
    int     iter=0;
    int     index;
    double  error=1;
    double  error_test=0;
    double  totalDiff=10;

    struct timespec start, now;

    double *Y       =(double *)calloc(MINI_BATCH_SIZE,sizeof(double));
    double *residual=(double *)calloc(MINI_BATCH_SIZE,sizeof(double));
    double *temp    =(double *)calloc(VECTOR_SIZE,sizeof(double));

    double **X=(double **)calloc(MINI_BATCH_SIZE,sizeof(double *));
    for(int i=0;i<MINI_BATCH_SIZE;i++)
        X[i]=(double *)calloc(VECTOR_SIZE,sizeof(double));

    double **jacobian=(double **)calloc(MINI_BATCH_SIZE,sizeof(double *));
    for(int i=0;i<MINI_BATCH_SIZE;i++)
        jacobian[i]=(double *)calloc(VECTOR_SIZE,sizeof(double));

    clock_gettime(CLOCK_MONOTONIC, &start);

    for(int i=0;i<MINI_BATCH_SIZE;i++)
    {
        X[i]=X_Full[i];
        Y[i]=Y_Full[i];
    }
    index=1;
    calculateResidual_SGD(residual,W,X,Y);

            //ITERASYON DONGUSU

    while(totalDiff>0.00002 && error>0.05 && iter<MAX_ITERATION)
    {
        for(int i=0;i<VECTOR_SIZE;i++)
        {
            allWValues[iter][i]=W[i];
        }
        totalDiff=0;
        printf("\nITERATION %d\n",iter+1);

            //YENI W HESAPLANMASI

        calculateJacobian_SGD(jacobian,W,X);
        for(int i=0;i<VECTOR_SIZE;i++)
        {
            temp[i]=0;
            for(int j=0;j<MINI_BATCH_SIZE;j++)
                temp[i]+=jacobian[j][i]*residual[j];

            temp[i]=temp[i]*2*eps;
        }

        for(int i=0;i<VECTOR_SIZE;i++)
        {
            W[i]=W[i]-temp[i];
            //printf("New W%d=%lf\n",i,W[i]);
            totalDiff+=fabs(temp[i]);
        }

        calculateResidual_SGD(residual,W,X,Y);
        error=0;
        for(int i=0;i<MINI_BATCH_SIZE;i++)
            error+=pow(residual[i],2);
        error/=MINI_BATCH_SIZE;
        totalDiff/=VECTOR_SIZE;
        error_test=testError(W,X_test,Y_test);

        errorTime[iter][0]=error;
        errorTime[iter][1]=error_test;
        clock_gettime(CLOCK_MONOTONIC, &now);
        errorTime[iter][2]=(now.tv_sec - start.tv_sec) * 1000.0+(now.tv_nsec - start.tv_nsec) / 1e6;
        if((index+1)*MINI_BATCH_SIZE>DATASET_SIZE)
        {
            index=0;
        }
        for(int i=0; i<MINI_BATCH_SIZE; i++)
        {
            int j=MINI_BATCH_SIZE*index+i;
            X[i]=X_Full[j];
            Y[i]=Y_Full[j];
        }
        //printf("TIME: %lf // ERROR: %lf\n",errorTime[iter][1],errorTime[iter][0]);
        index++;
        iter++;

        printf("\nTrain Error: %lf\nTest Error: %lf\nCurrent Difference: %lf\n---------------------\n---------------------\n---------------------\n",error,error_test,totalDiff);
    }
    if(error<=0.05)
        printf("Function converged with %d iterations! Last error rate:%lf\n",iter,error);
    else
        printf("Function did not converge! Last error rate:%lf",error);
    errorTime[iter]=NULL;
    for(int i=0;i<VECTOR_SIZE;i++)
    {
        allWValues[iter][i]=W[i];
    }
    allWValues[iter+1]=NULL;
}

void arrayShuffler(double **X, double *Y)
{
    double *tempX;
    double tempY;
    for(int i=DATASET_SIZE-1;i>=0;i--)
    {
        int randomIndex = rand()%(i+1);
        tempX=X[i];
        X[i]=X[randomIndex];
        X[randomIndex]=tempX;

        tempY=Y[i];
        Y[i]=Y[randomIndex];
        Y[randomIndex]=tempY;
    }
}

void ADAM(double *W, double **X, double *Y, double alpha, double **errorTime, double **allWValues, double **X_test, double *Y_test)
{
    int    index=1;
    int    t=0;
    struct timespec start, now;
    double B1=0.9;
    double B2=0.999;
    double error=1;
    double error_test=0;
    double totalDiff=10;
    double productResult=0;

    double *gt      =(double *)calloc(VECTOR_SIZE,sizeof(double));
    double *residual=(double *)calloc(DATASET_SIZE,sizeof(double));
    double *mt      =(double *)calloc(VECTOR_SIZE,sizeof(double));
    double *vt      =(double *)calloc(VECTOR_SIZE,sizeof(double));
    double *mt_hat  =(double *)calloc(VECTOR_SIZE,sizeof(double));
    double *vt_hat  =(double *)calloc(VECTOR_SIZE,sizeof(double));

    clock_gettime(CLOCK_MONOTONIC, &start);

    for(int i=0;i<VECTOR_SIZE;i++)
    {
        mt[i]=0;
        vt[i]=0;
    }
    calculateResidual(residual,W,X,Y);
    error=0;
    for(int i=0;i<DATASET_SIZE;i++)
        error+=pow(residual[i],2);
    error/=DATASET_SIZE;
    errorTime[0][0]=error;
    errorTime[0][1]=0.0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while(totalDiff>0.00001 && error>0.05 && t<MAX_ITERATION)
    {
        for(int i=0;i<VECTOR_SIZE;i++)
        {
            allWValues[t][i]=W[i];
        }
        t++;
        printf("ITERATION %d\n",t);
        totalDiff=0;
        double currentResidue=Y[index]-tanh(vectorProduct(W,X[index],VECTOR_SIZE));
        for(int i=0;i<VECTOR_SIZE;i++)
        {
            //gt hesaplamasi
            productResult=0;
            for(int j=0;j<VECTOR_SIZE;j++)
            {
                if(j!=i)
                {
                    productResult+=W[j]*X[index][j];
                }
            }
            gt[i]=currentResidue *X[index][i] *(-2)* pow( 1/cosh(X[index][i]*W[i]+productResult) , 2);

            //mt, vt ve w hesaplari
            mt[i]=B1*mt[i]+(1-B1)*gt[i];
            vt[i]=B2*vt[i]+(1-B2)*gt[i]*gt[i];
            mt_hat[i]=mt[i]/(1-pow(B1,t));
            vt_hat[i]=vt[i]/(1-pow(B2,t));

            double tmp=W[i]-alpha*mt_hat[i]/(sqrt(vt_hat[i])+0.00000001); //0'a bolmeyi engellemek icin
            totalDiff+=fabs(W[i]-tmp);
            //printf("%lf\n",totalDiff);
            W[i]=tmp;
        }

        calculateResidual(residual,W,X,Y);
        error=0;
        for(int i=0;i<DATASET_SIZE;i++)
            error+=pow(residual[i],2);
        error/=DATASET_SIZE;
        totalDiff/=VECTOR_SIZE;
        error_test=testError(W,X_test,Y_test);
        errorTime[t][0]=error;
        errorTime[t][1]=error_test;
        clock_gettime(CLOCK_MONOTONIC, &now);
        errorTime[t][2]=(now.tv_sec - start.tv_sec) * 1000.0+(now.tv_nsec - start.tv_nsec) / 1e6;
        //printf("TIME: %lf // ERROR: %lf\n",errorTime[t][1],errorTime[t][0]);
        index++;
        if(index>=DATASET_SIZE)
        {
            index=0;
        }

        printf("\nTrain Error: %lf\nTest Error: %lf\nCurrent Difference: %lf\n---------------------\n---------------------\n---------------------\n",error,error_test,totalDiff);

    }
    if(error<=0.05)
        printf("Function converged with %d iterations! Last error rate:%lf\n",t,error);
    else
        printf("Function did not converge! Last error rate:%lf",error);
    errorTime[t+1]=NULL;
    for(int i=0;i<VECTOR_SIZE;i++)
    {
        allWValues[t][i]=W[i];
    }
    allWValues[t+1]=NULL;
}

void saveError(double **errorTime, int func)
{
    FILE* file;
    if(func==0)//gd
    {
        file=fopen("gdError2701.txt","w");
    }
    else if(func==1)//sgd
    {
        file=fopen("sgdError1911.txt","w");
    }
    else//adam
    {
        file=fopen("adamError0.txt","w");
    }
    int i=0;
    while(errorTime[i])
    {
        fprintf(file,"%lf,%lf,%lf\n",errorTime[i][0],errorTime[i][1],errorTime[i][2]);
        i++;
    }
}

void saveW(double **W, int func)
{
    FILE* file;
    if(func==0)//gd
    {
        file=fopen("gdW2701.txt","w");
    }
    else if(func==1)//sgd
    {
        file=fopen("sgdW1911.txt","w");
    }
    else//adam
    {
        file=fopen("adamW0.txt","w");
    }
    int i=0;
    while(W[i])
    {
        fprintf(file,"%lf",W[i][0]);
        for(int j=1;j<VECTOR_SIZE;j++)
        {
            fprintf(file,",%lf",W[i][j]);
        }
        fprintf(file,"\n");
        i++;
    }
}


int main() {
    double stepSize=0.003;

    int testResult=0;

    double *W           =(double *)calloc(VECTOR_SIZE    ,sizeof(double));
    double *Y           =(double *)calloc(DATASET_SIZE   ,sizeof(double));
    double *Y_test      =(double *)calloc(TESTSET_SIZE   ,sizeof(double));

    double **X          =(double **)calloc(DATASET_SIZE   ,sizeof(double *));
    double **X_test     =(double **)calloc(TESTSET_SIZE   ,sizeof(double *));
    double **allWValues =(double **)calloc(MAX_ITERATION+1,sizeof(double *));
    double **errorTime  =(double **)calloc(MAX_ITERATION+1,sizeof(double *));
    unsigned char **imageMatrix=(char **)calloc(IMAGE_SIZE,sizeof(char *));

    for(int i=0;i<IMAGE_SIZE;i++)
    {
        imageMatrix[i]=(char *)calloc(IMAGE_SIZE,sizeof(char));
    }
    for(int i=0;i<DATASET_SIZE;i++)
    {
        X[i]=(double *)calloc(VECTOR_SIZE,sizeof(double));
    }
    for(int i=0;i<TESTSET_SIZE;i++)
    {
        X_test[i]=(double *)calloc(VECTOR_SIZE,sizeof(double));
    }
    for(int i=0;i<MAX_ITERATION+1;i++)
    {
        allWValues[i]=(double *)calloc(VECTOR_SIZE,sizeof(double));
    }
    for(int i=0;i<MAX_ITERATION+1;i++)
    {
        errorTime[i]=(double *)calloc(3,sizeof(double));
    }

    //Gorselleri matrise cevirip ardindan 0-1 araligina normalize ederek n^2+1 uzunluklu bir vektore kaydet
    for(int i=0 ; i<DATASET_SIZE ; i++)
    {
        load_pgm(imageName(i,1), imageMatrix);
        X[i][0]=1; //bias
        for(int j=0;j<IMAGE_SIZE;j++)
        {
            for(int k=0;k<IMAGE_SIZE;k++)
            {
                X[i][j*IMAGE_SIZE+k+1]=imageMatrix[j][k];
                X[i][j*IMAGE_SIZE+k+1]/=255;
            }
        }
        printf("%d\n",i);
    }

    for(int i=0;i<DATASET_SIZE/2;i++)
    {
        Y[i]=1;
        Y[i+(DATASET_SIZE/2)]=-1;
    }

    printf("\n\nTEST SET:\n");
    for(int i=0 ; i<TESTSET_SIZE ; i++)
    {
        load_pgm(imageName(i,0), imageMatrix);
        X_test[i][0]=1; //bias
        for(int j=0;j<IMAGE_SIZE;j++)
        {
            for(int k=0;k<IMAGE_SIZE;k++)
            {
                X_test[i][j*IMAGE_SIZE+k+1]=imageMatrix[j][k];
                X_test[i][j*IMAGE_SIZE+k+1]/=255;
            }
        }
        printf("%d\n",i);
    }
    for(int i=0;i<TESTSET_SIZE/2;i++)
    {
        Y_test[i]=1;
        Y_test[i+(TESTSET_SIZE/2)]=-1;
    }

    arrayShuffler(X,Y);

    srand(2702);
    for(int i=0;i<VECTOR_SIZE;i++)
    {
        W[i] = (rand() % 100) / 5000.0; // [0, 0.05] arasi degerler
    }
    gradientDescent(W,X,Y,STEP_SIZE,errorTime,allWValues,X_test,Y_test);
    saveError(errorTime,0);
    saveW(allWValues,0);

    testResult=test(W,X_test,Y_test);
    printf("\n\nLAST RESULT:\n%d CORRECT GUESSES\n%d WRONG GUESSES",testResult,TESTSET_SIZE-testResult);
    return 0;
}
