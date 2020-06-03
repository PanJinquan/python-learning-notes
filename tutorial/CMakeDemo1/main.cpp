#include <iostream>
#include <fstream>
#include "myhello.h"
using namespace std;

int main()
{
  int row=2;
  int col=5;
  double fnum[row][col] = {0};
  ifstream in("bin/data.bin", ios::in | ios::binary);
  in.read((char *) &fnum, sizeof fnum);
  cout << in.gcount() << " bytes read\n";
  // show values read from file
  for(int i=0; i<row; i++){
      for(int j=0;j<col;j++){
            cout << fnum[i][j] << ",";
      }
       std::cout<<endl;
  }
  in.close();
  return 0;
}