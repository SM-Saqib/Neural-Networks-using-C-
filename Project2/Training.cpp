#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>



using namespace std;
//const string weights = "weights/b_hidden_layer";
//const string image = "mnist/t10k-images.idx3-ubyte";
const string testing_image_fn = "mnist/train-images.idx3-ubyte";
const string model_fn = "model-neural-network.dat";
const string model_fn1 = "model-neural-network1.dat";
const string testing_label_fn = "mnist/train-labels.idx1-ubyte";


// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

//double X[784];


const int width = 28;
const int height = 28;
const int n1 = width * height;
const int n2 = 128;
const int n3 = 10;
double delta1[784][128];
double delta2[128][10];
//double d[width+1][height+1];
const int nTraining = 1400;//60000;

int FeedForward(double *Xi, double Wij[][128], double Wjk[][10], double *outputj, double *outputk, double *DesOut);

int inputs(double *X, double *desired);
void write_matrix(string file_name, double w1[][128], double w2[][10], int n1, int n2, int n3);

void sigmoid(double *arr, int len);
double sqerror(double *tk, double *zk, int len);


//for a [][128] neuron layer

void FeedForward100(double *input, double *output, double weight[][128], int sizeInput, int sizeOutput);
double FFstep100(double *, double weight[][128], int, int);





// functions for col 10 arrays
void FeedForward10(double *input, double *output, double weight[][10], int sizeInput, int sizeOutput);
double FFstep10(double *, double weight[][10], int, int);

void w8Tozero(double w1[][128], double w2[][10]);
void w8Tohalf(double w1[][128], double w2[][10]);
void w8input(double w1[][128], double w2[][10], string file_name);

void ArrTozero(double *w, int size);

//Backpropagation part

double Dsigmoi(double x);
void SenstivO(double *tk, double *zk, double *SenO, int siz);
void SenstivH10(double *y, double weight[][10], int rowWeights, double *senThis, double *senNext);
void inputW8Update(double *X, double *SenH, double weight[][10], int rowWeight, int Col ,double learningRate);
void inputW8Update128(double *X, double *SenH, double weight[][128], int rowWeight, int Col, double learningRate);

int main()
{
	//report.open(report_fn.c_str(), ios::out);
	image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
	label.open(testing_label_fn.c_str(), ios::in | ios::binary); // Binary label file


	char number;
	for (int i = 1; i <= 16; ++i) {
		image.read(&number, sizeof(char));
	}
	for (int i = 1; i <= 8; ++i) {
		label.read(&number, sizeof(char));
	}

	double Wij[785][128];
	double outputj[128];
	double Xi[784];
	double Wjk[128][10];
	double outputk[10];
	double sensO[10];
	double sensH[128];
	double DesOut[10];
	w8Tohalf(Wij, Wjk);
	w8Tozero(delta1, delta2);
	//w8input(Wij, Wjk, model_fn);

	for (int sample = 1; sample <= nTraining; ++sample)
	{
		/*double *Xi;
		double outputj[128];
		double outputk[10];
		double *DesOut;

		Xi = new double[785];
		DesOut = new double[10];*/


		ArrTozero(Xi, 784);
		ArrTozero(outputj, 128);
		ArrTozero(outputk, 10);
		ArrTozero(DesOut, 10);
		ArrTozero(sensO, 10);
		ArrTozero(sensH, 128);
		

		cout << "Sample " << sample << endl;
		int LabelNum=inputs(Xi, DesOut);
		


		int predict = FeedForward(Xi, Wij, Wjk, outputj, outputk, DesOut);
		cout << "predict  before training " << predict << "   " << endl<<endl;

		//backpropagationPart
		int iteration = 0;
		while ((iteration<512))
		{
			
			iteration++;
				SenstivO(DesOut, outputk, sensO, 10);
				SenstivH10(outputj, Wjk, 10, sensH, sensO);
				inputW8Update(outputj, sensO, Wjk, 128, 10,1e-3);
				inputW8Update128(Xi, sensH, Wij, 784, 128, 1e-3);
				predict= FeedForward(Xi, Wij, Wjk, outputj, outputk, DesOut);
				//cout << iteration << endl;
				double error=sqerror(DesOut, outputk, 10);
				if(error<(1e-3))
				{
					break;
				}
			}
		cout << "predict after training  " << predict << "   " << endl;
		cout << "Label   " << LabelNum << "   " << endl<<endl<<endl;

		
	
	}

	write_matrix(model_fn1,Wij,Wjk,784,128,10);

	report.close();
	image.close();
	label.close();

	return 0;
}



void w8Tozero(double w1[][128], double w2[][10])
{

	//input layer to hidden layer

	for (int i = 1; i <= n1; ++i)
	{
		for (int j = 1; j <= n2; ++j)
		{
			w1[i][j] = 0;

		}
	}
	// hidden to output

	for (int i = 1; i <= n2; ++i)
	{
		for (int j = 1; j <= n3; ++j)
		{
			w2[i][j] = 0;

		}
	}

}

//weightInitialization


void w8Tohalf (double w1[][128], double w2[][10])
{

	//input layer to hidden layer

	for (int i = 1; i <= n1; ++i)
	{
		for (int j = 1; j <= n2; ++j)
		{
			w1[i][j] = (double)(rand() % 10 + 1) / (10 * n2);

		}
	}
	// hidden to output

	for (int i = 1; i <= n2; ++i)
	{
		for (int j = 1; j <= n3; ++j)
		{
			int sign = rand() % 2;

			// Another strategy to randomize the weights - quite good
			// w2[i][j] = (double)(rand() % 6) / 10.0;

			w2[i][j] = (double)(rand() % 10 + 1) / (10.0 * n3);
			if (sign == 1) {
				w2[i][j] = -w2[i][j];
			}

		}
	}

}

//const string
void w8input(double w1[][128], double w2[][10], string file_name)
{
	ifstream file(file_name.c_str(), ios::in);
	//input layer to hidden layer

	for (int i = 1; i <= n1; ++i)
	{
		for (int j = 1; j <= n2; ++j)
		{
			file >> w1[i][j];
			//cout<<w1[i][j]<<endl;
		}
	}
	// hidden to output

	for (int i = 1; i <= n2; ++i)
	{
		for (int j = 1; j <= n3; ++j)
		{
			file >> w2[i][j];

		}
	}
	file.close();
}

void sigmoid(double *arr, int len)
{
	for (int i = 0; i < len; i++)
	{
		arr[i] = (double)(1.0 / (1.0 + exp(-arr[i])));

	}
}



void FeedForward100(double *input, double *output, double weight[][128], int sizeInput, int sizeOutput)
{
	for (int i = 1; i < sizeInput; i++)
	{
		for (int j = 1; j < sizeOutput; j++)
			output[j] += (input[i] * weight[i][j]);

	}
}



double FFstep100(double *input, double weight[][128], int sizeIn, int j)
{
	double Y = 0;
	for (int i = 1; i < sizeIn; i++)
	{
		Y += input[i] * weight[i][j];
	}
	return Y;
}

void FeedForward10(double *input, double *output, double weight[][10], int sizeInput, int sizeOutput)
{
	for (int i = 1; i < sizeInput; i++)
	{

		for (int j = 1; j < sizeOutput; j++)
		{

			output[j] = output[j] + (input[i] * weight[i][j]);

		}
	}

}


/*double FFstep10(double *input, double weight[][10], int sizeIn, int j)
{
	double Y = 0;
	for (int i = 0; i < sizeIn; i++)
	{
		double weightd = weight[i][j];
		Y += input[i] * weightd;
	}
	return Y;
}*/

double sqerror(double *tk, double *zk, int len)
{
	double sum1 = 0;
	for (int i = 1; i < len; i++)
	{
		sum1 = sum1 + ((zk[i] - tk[i])*(zk[i] - tk[i]));

	}

	sum1 = (double)(sum1 * 0.5);
	return sum1;
}

int inputs(double *X, double *desired) {
	// Reading image
	char number;
	double d[width + 1][height + 1];
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			image.read(&number, sizeof(char));
			if (number == 0) {
				d[i][j] = 0;
			}
			else {
				d[i][j] = 1;
			}
		}
	}

	cout << "Image:" << endl;
	for (int j = 1; j < height; ++j) {
		for (int i = 1; i < width; ++i) {
			//cout << d[i][j];
		}
		//cout << endl;
	}

	for (int j = 1; j <= height; j++) {
		for (int i = 1; i <= width; i++) {
			int pos = i + (j - 1) * width;
			X[pos] = d[i][j];
		}
	}

	// Reading label
	label.read(&number, sizeof(char));
	for (int i = 1; i < n3; ++i) {
		desired[i] = 0.0;
	}
	desired[number + 1] = 1.0;
	int LabelNo = (int)(number);
	cout << "Label before: " <<LabelNo << endl;
	return LabelNo;
}




void ArrTozero(double *w, int size)
{

	//input layer to hidden layer

	for (int i = 0; i <= size; ++i)
	{

		w[i] = 0;


	}
}





void SenstivO(double *tk, double *zk, double *SenO, int siz)
{
	// tk=desired output  // zk=calculated



	for (int i = 0; i < siz; i++)
	{
		double fnet = Dsigmoi(zk[i]);// derivative of the activation function
		SenO[i] = (tk[i] - zk[i])*fnet;
		// cout<<"sensitivity "<<SenO[i]<<endl;

	}
}


double Dsigmoi(double x)
{
	return x * (1 - x);

}



void inputW8Update(double *X, double *SenH, double weight[][10], int rowWeight,int Col ,double learningRate)
{



	for (int i = 0; i < rowWeight; i++)
	{
		for (int j = 0; j < Col; j++)
		{

			delta2[i][j] = (learningRate * SenH[j] * X[i] + (0.9*delta2[i][j]));
			weight[i][j] = weight[i][j] + delta2[i][j];

		}
	}

}

void inputW8Update128(double *X, double *SenH, double weight[][128], int rowWeight, int Col, double learningRate)
{



	for (int i = 0; i < rowWeight; i++)
	{
		for (int j = 0; j < Col; j++)
		{


			delta1[i][j] = (learningRate * SenH[j] * X[i] + (0.9*delta1[i][j]));
				weight[i][j] = weight[i][j] + delta1[i][j];


		}
	}

}

void SenstivH10(double *y, double weight[][10], int rowWeights, double *senThis, double *senNext) // calculates the sensitivity for individual hidden nueron.
{

	int sizeOut = 128;







	for (int k = 0; k < sizeOut; k++)
	{
		int j = 0;
		double Sum = 0;

		for (int i = 0; i < rowWeights; i++)
		{

			Sum = Sum + weight[k][i] * senNext[i];


			//j++;

		}


		senThis[k] = y[k] * (1 - y[k])*Sum;
		//cout<<"index  ["<<k<<"]  "<<senThis[k];
	}

}

void w8RandInit10(double weight[][10], int rows)
{
	int k = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			weight[i][j] = 1;//(double)((5)/(j+2));
			//cout<<"  index "<< ++k<<"   "<<weight[i][j]<<endl;
		}
	}

}

void w8RandInit100(double weight[][100], int rows)
{
	int k = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < 100; j++)
		{
			weight[i][j] = 0.001;//(double)((3)/(j+2));
			//cout<<"  index "<< ++k<<"   "<<weight[i][j]<<endl;
		}
	}

}


int FeedForward(double *Xi,double Wij[][128],double Wjk[][10],double *outputj, double *outputk,double *DesOut)
{
	FeedForward100(Xi, outputj, Wij, 784, 128);
	sigmoid(outputj, 128);



	FeedForward10(outputj, outputk, Wjk, 128, 10);
	sigmoid(outputk, 10);



	double error = sqerror(DesOut, outputk, 10);

	int predict = 1;
	for (int i = 2; i <= n3; ++i) {
		if (outputk[i] > outputk[predict]) {
			predict = i;
			//cout<<"predict   "<<predict<<"   "<<endl<<"label  "<<label<<endl;
		}



	}
	predict--;
	return predict;
}

void write_matrix(string file_name,double w1[][128],double w2[][10],int n1,int n2,int n3)
{
	ofstream file(file_name.c_str(), ios::out);

	// Input layer - Hidden layer
	for (int i = 1; i <= n1; ++i) {
		for (int j = 1; j <= n2; ++j) {
			file << w1[i][j] << " ";
		}
		file << endl;
	}

	// Hidden layer - Output layer
	for (int i = 1; i <= n2; ++i) {
		for (int j = 1; j <= n3; ++j) {
			file << w2[i][j] << " ";
		}
		file << endl;
	}

	file.close();
}
