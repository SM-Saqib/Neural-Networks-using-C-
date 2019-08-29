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
const string testing_image_fn = "mnist/t10k-images.idx3-ubyte";
const string model_fn = "model-neural-network.dat";
const string testing_label_fn = "mnist/t10k-labels.idx1-ubyte";


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

//double d[width+1][height+1];
const int nTraining = 20;//60000;


void inputs(double *X, double *desired);

void sigmoid(double *arr, int len);
double sqerror(double *tk, double *zk, int len);


//for a [][128] neuron layer

void FeedForward100(double *input, double *output, double weight[][128], int sizeInput, int sizeOutput);
double FFstep100(double *, double weight[][128], int, int);





// functions for col 10 arrays
void FeedForward10(double *input, double *output, double weight[][10], int sizeInput, int sizeOutput);
double FFstep10(double *, double weight[][10], int, int);

void w8Tozero(double w1[][128], double w2[][10]);
void w8input(double w1[][128], double w2[][10], string file_name);

void ArrTozero(double *w, int size);




int main()
{
	//report.open(report_fn.c_str(), ios::out);
	image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
	//label.open(testing_label_fn.c_str(), ios::in | ios::binary); // Binary label file


	char number;
	for (int i = 1; i <= 16; ++i) {
		image.read(&number, sizeof(char));
	}
	for (int i = 1; i <= 8; ++i) {
		//label.read(&number, sizeof(char));
	}

	double Wij[784][128];
	//double outputj[128];
	//double Xi[784];
	double Wjk[128][10];
	//double outputk[10];
	double sensO[10];
	double sensH[128];
	//double DesOut[10];
	w8Tozero(Wij, Wjk);
	w8input(Wij, Wjk, model_fn);

	for (int sample = 1; sample <= nTraining; ++sample)
	{
		double Xi[784];
		double outputj[128];
		double outputk[10];
		double DesOut[10];
		ArrTozero(Xi, 784);
		ArrTozero(outputj, 128);
		ArrTozero(outputk, 10);
		ArrTozero(DesOut, 10);
		
		inputs(Xi, DesOut);
		cout << "Sample " << sample << endl;

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
			
			////<<"label  "<<label<<endl;
			
		}
		predict--;
		cout << "predict   " << predict << "   " << endl;
	}

	image.close();
	system("pause");
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
		output[i] = FFstep100(input, weight, sizeOutput, i);

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

void inputs(double *X, double *desired) {
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
			cout << d[i][j];
		}
		cout << endl;
	}

	for (int j = 1; j <= height; j++) {
		for (int i = 1; i <= width; i++) {
			int pos = i + (j - 1) * width;
			X[pos - 1] = d[i][j];
		}
	}

	// Reading label
	label.read(&number, sizeof(char));
	for (int i = 1; i < n3; ++i) {
		desired[i] = 0.0;
	}
	desired[number] = 1.0;

	cout << "Label: " << (int)(number) << endl;
}




void ArrTozero(double *w,int size)
{

	//input layer to hidden layer

	for (int i = 0; i <= size; ++i)
	{
		
			w[i] = 0;
			

		}
	}