#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


float sigmoid(float x){
    float sig = 1/(1+exp(-x));
    return sig;
}

int main() {
    
    
    //Fast forward
    int i=0, j=0, k=0, update=0;
    float outputa1[2]={0}, outputa2[1]={0};
    float  inputx[4][2]={-0.5,-0.5,
        -0.5,1,
        1,-0.5,
        1,1};
    
    float inputw1[2][2]={0.9,0.1,0.3,0.4}, inputw2[1][2]={0.9,-0.1}, Ans[4]={0, 1, 1, 0}, E=0;
    
    //BP(learning rate=1)
    float delta1[2]={0}, delta2[1]={0}, threshold=0.1, updatew1[2][2]={0}, updatew2[1][2]={0};
    
    srand(time(NULL));
    do{
        
        //        update=0;
        
        //        for(k=0; k<4; k++){
        
        
        k = rand() % 4;
//        k=3;
//        printf("%d\n",k);
        
        for(j=0; j<2; j++){
            outputa1[j] = 0;
            for(i=0; i<2; i++){
                outputa1[j] = outputa1[j] + inputw1[j][i] * inputx[k][i];
                //                                    printf("%f\n", outputa1[j]);
            }
            outputa1[j] = sigmoid(outputa1[j]);
//            printf("outputa1[%d]=%f\n", j, outputa1[j]);
        }
        
        for(j=0; j<1; j++){
            outputa2[j] = 0;
            for(i=0; i<2; i++){
                outputa2[j] = outputa2[j] + inputw2[j][i] * outputa1[i];
            }
            outputa2[j] = sigmoid(outputa2[j]);
//            printf("outputa2[%d]=%f\n", j, outputa2[j]);
            
//            E = pow(Ans[k] - outputa2[j], 2);
//            printf("E=%f\n", E);
        }
        
        printf("%d, outputa2=%f\n",k, outputa2[0]);
        
        
        
        //Back Path
        //            if( fabs(outputa2-Ans[k]) > 0.1 ){
        //                printf("%f\n", outputa2-Ans[k]);
        
        for(j=0; j<1; j++){
            delta2[j] = (Ans[k] - outputa2[j]) * outputa2[j] * (1-outputa2[j]);
//           printf("delta2[%d]=%f\n", j, delta2[j]);
        }
        
        for(j=0; j<1; j++){
            for(i=0; i<2; i++){
//               printf("inputw2[%d][%d]=%f\n", j, i, inputw2[j][i]);
            }
        }
        
        for(i=0; i<2; i++){
            delta1[i] = 0;
            for(j=0; j<1; j++){
                delta1[i] = delta1[i] + inputw2[j][i] * delta2[j] * outputa1[i] * (1-outputa1[i]);
//              printf("delta1[%d]=%f\n", i, delta1[i]);
            }
        }
        
        for(j=0; j<1; j++){
            for(i=0; i<2; i++){
                updatew2[j][i] = delta2[j] * outputa1[i];
                inputw2[j][i] = inputw2[j][i] + updatew2[j][i];
//                printf("updatew2[%d][%d]=%f\n", j, i, updatew2[j][i]);
//                printf("neww2[%d][%d]=%f\n", j, i, inputw2[j][i]);
            }
        }
        
        for(j=0; j<2; j++){
            for(i=0; i<2; i++){
                updatew1[j][i] = delta1[j] * inputx[k][i];
                inputw1[j][i] = inputw1[j][i] + updatew1[j][i];
//                printf("updatew1[%d][%d]=%f\n", j, i, updatew1[j][i]);
//                printf("neww1[%d][%d]=%f\n", j, i, inputw1[j][i]);
            }
        }
        
        
//        update++;
    }while(update<=1000000);
    
    
    
     
    return 0;
}
