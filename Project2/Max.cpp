#include <iostream>
using namespace std;

float maxFun(float arr[16])
{
	float max=0;
	for(int i=0;i<16;i++)
		if (max < arr[i])
		{
			max = arr[i];
		}
	return max;
}
int main()
{
	float arr[16] = { 1,2,3,4,15,16,7,8,9,5,6,7,8,9,10,2 };
	cout<<"the answer" <<(maxFun(arr));
	system("pause");

}