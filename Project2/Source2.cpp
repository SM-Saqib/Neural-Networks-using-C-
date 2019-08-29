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


const int width = 28;
const int height = 28;
const int n1 = width * height;
const int n2 = 128;
const int n3 = 10;

float d[width + 1][height + 1];
const int nTraining = 2;//60000;


void inputs(float *X, float *desired);

void sigmoid(float *arr, int len);
float sqerror(float *tk, float *zk, int len);


//for a [][100] neuron layer

void FeedForward100(float *input, float *output, float weight[][100], int sizeInput, int sizeOutput);
float FFstep100(float *, float weight[][100], int, int);





// functions for col 10 arrays
void FeedForward10(float *input, float *output, float weight[][10], int sizeInput, int sizeOutput);
float FFstep10(float *, float weight[][10], int, int);







//const string 
void w8input(float w1[][100],float w2[][10],string file_name)
{
	ifstream file(file_name.c_str(), ios::in);
	//input layer to hidden layer

	for (int i = 1; i <= n1; i++)
	{
		for (int j = 1; j <= n2; j++)
		{
			file >> w1[i][j];

		}
	}
	// hidden to output

	for (int i = 1; i <= n2; i++)
	{
		for (int j = 1; j <= n3; j++)
		{
			file >> w2[i][j];

		}
	}
	file.close();
}

void sigmoid(float *arr, int len)
{
	for (int i = 0; i < len; i++)
	{
		arr[i] = (float)(1.0 / (1.0 + exp(-(arr[i]))));
	}
}



void FeedForward100(float *input, float *output, float weight[][100], int sizeInput, int sizeOutput)
{
	for (int i = 0; i < sizeOutput; i++)
	{
		output[i] = FFstep100(input, weight, sizeInput, i);

	}

}

float FFstep100(float *input, float weight[][100], int sizeIn, int j)
{
	float Y = 0;
	for (int i = 0; i < sizeIn; i++)
	{
		Y += input[i] * weight[i][j];
	}
	return Y;
}

void FeedForward10(float *input, float *output, float weight[][10], int sizeInput, int sizeOutput)
{
	for (int i = 0; i < sizeOutput; i++)
	{
		output[i] = FFstep10(input, weight, sizeInput, i);

	}

}


float FFstep10(float *input, float weight[][10], int sizeIn, int j)
{
	float Y = 0;
	for (int i = 0; i < sizeIn; i++)
	{
		Y += input[i] * weight[i][j];
	}
	return Y;
}

float sqerror(float *tk, float *zk, int len)
{
	float sum1 = 0;
	for (int i = 1; i < len; i++)
	{
		sum1 = sum1 + ((zk[i] - tk[i])*(zk[i] - tk[i]));

	}

	sum1 = (float)(sum1 * 0.5);
	return sum1;
}

void inputs(float *X, float *desired) {
	// Reading image
	char number = 0;
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
/*	label.read(&number, sizeof(char));
	for (int i = 0; i < n3; ++i) {
		desired[i] = 0.0;
	}
	desired[number] = 1.0;

	cout << "Label: " << (int)(number) << endl;*/
}


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

	float Wij[784][100];
	float outputj[100];
	float Xi[784];
	float Wjk[100][10];
	float outputk[10];
	float sensO[10];
	float sensH[100];
	float DesOut[10];

	for (int sample = 1; sample <= nTraining; ++sample)
	{

		cout << "Sample " << sample << endl;

		FeedForward100(Xi, outputj, Wij, 784, 100);
		sigmoid(outputj, 100);



		FeedForward10(outputj, outputk, Wjk, 100, 10);
		sigmoid(outputk, 10);



		float error = sqerror(DesOut, outputk, 10);
	}

	return 0;
}
