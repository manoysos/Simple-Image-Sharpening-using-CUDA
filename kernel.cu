
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct PGMstruct
{
	int maxGrey;
	int width;
	int height;
	int **matrix;
	int **matrixTr;
};

typedef struct PGMstruct PGMImage;


__global__ void gpuSharpenImg3x3(int * d_matrix, int * d_matrixTr, int width, int height) //Kernel για την εφαρμογή του φίλτρου 3x3 χρησιμοποιώντας την GPU.
{
	
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int upL = index - (width + 1);
	int upM = index - (width);
	int upR = index - (width - 1);
	int left = index - 1;
	int right = index + 1;
	int downL = index + (width - 1);
	int downM = index + (width);
	int downR = index + (width + 1);

	
	//Υπολογισμός νέας τιμής μόνο για τα στοιχεία που δεν βρίσκονται στις πρώτες και τελευταίες γραμμές και στήλες.
	if ((index > width - 1) && (index < (width*height) - width) && (index%width != 0) && (index%width != width - 1))
	{
		d_matrixTr[index] = (-1)*d_matrix[upL] + (-1)*d_matrix[upM] + (-1)*d_matrix[upR] + (-1)*d_matrix[left] + (-1)*d_matrix[right] + (-1)*d_matrix[downL] + (-1)*d_matrix[downM] + (-1)*d_matrix[downR] + (9 * d_matrix[index]);
		if (d_matrixTr[index]>255)
		{
			d_matrixTr[index] = 255;
		}
		if (d_matrixTr[index]<0)
		{
			d_matrixTr[index] = 0;
		}
	}  
	
	__syncthreads();
}


__global__ void gpuSharpenImg5x5(int*d_matrix, int*d_matrixTr, int width, int height)  //Kernel για την εφαρμογή του φίλτρου 5x5 χρησιμοποιώντας την GPU.
{
	/*
	______________________________________
	
	|upUL  |upULM  |upUM  |upURM  |upUR  |
	|upL   |upLM   |upM   |upRM   |upR   |
	|leftL |left   |index |right  |rightR|
	|downL |downLM |downM |downRM |downR |
	|downDL|downDLM|downDM|downDRM|downDR|
	______________________________________
	
	*/
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int	upUL = index - 2 * width - 2;
	int	upULM = index - 2 * width - 1;
	int	upUM = index - 2 * width;
	int	upURM = index - 2 * width + 1;
	int	upUR = index - 2 * width + 2;
	int	upL = index - width - 2;
	int	upLM = index - width - 1;
	int	upM	 = index - width;
	int	upRM = index - width + 1;
	int	upR = index - width + 2;
	int	leftL = index - 2;
	int	left = index - 1;
	int	right = index + 1;
	int	rightR = index + 2;
	int	downL = index + width - 2;
	int	downLM = index + width - 1;
	int downM = index + width;
	int downRM = index + width + 1;
	int downR = index + width + 2;
	int downDL = index + 2 * width - 2;
	int downDLM = index + 2 * width - 1;
	int downDM = index + 2 * width;
	int downDRM = index + 2 * width + 1;
	int downDR = index + 2 * width + 2;

	//Υπολογισμός νέας τιμής μόνο για τα στοιχεία που δεν βρίσκονται στις δύο πρώτες και δύο τελευταίες γραμμές και στήλες.
	if ((index > 2 * width - 1) && (index < (width*height) - 2 * width) && (index%width != 0) && (index%width != width - 1) && (index%width != 1) && (index%width != width - 2))
	{
		d_matrixTr[index] = ((-1)*(d_matrix[upUL] + d_matrix[upULM] + d_matrix[upUM] + d_matrix[upURM] +
			d_matrix[upUR] + d_matrix[upL] + d_matrix[upR] + d_matrix[leftL] +
			d_matrix[rightR] + d_matrix[downL] + d_matrix[downR] + d_matrix[downDL] +
			d_matrix[downDLM] + d_matrix[downDM] + d_matrix[downDRM] + d_matrix[downDR])
			+ 2 * (d_matrix[upLM] + d_matrix[upM] + d_matrix[upRM] +
			d_matrix[left] + d_matrix[right] + d_matrix[downLM] +
			d_matrix[downM] + d_matrix[downRM]) + 8 * d_matrix[index]) / 8;
		if (d_matrixTr[index]>255)
		{
			d_matrixTr[index] = 255;
		}
		if (d_matrixTr[index]<0)
		{
			d_matrixTr[index] = 0;
		}
	}
	__syncthreads();
}

//Συνάρτηση για την απελευθέρωση της μνήμης που έχει κατανεμηθεί για κάποιον δυναμικό πίνακα 2 διαστάσεων.
void deallocate_dynamic_matrix(int **matrix, int row)
{
	int i;

	for (i = 0; i < row; ++i)
		free(matrix[i]);
	free(matrix);
}

//Συνάρτηση που ανοίγει και διαβάζει το αρχείο pgm που έχει καθοριστεί από τον χρήστη.
int getPGM(const char *flnm, PGMImage *pgm)
{
	FILE *pgmFile;
	char ch;
	int type, col, row;
	int ch_int;
	pgmFile = fopen(flnm, "rb");
	//Μήνυμα σφάλματος αν δεν μπορεί να ανοιχτεί το αρχείο που έχει καθορίσει ο χρήστης.
	if (!pgmFile)
	{
		perror("Cannot open file");

		exit(EXIT_FAILURE);
	}

	printf("\nReading image file: %s\n", pgmFile);

	ch = getc(pgmFile);
	if (ch != 'P')
	{
		printf("ERROR(1): Not valid pgm file type\n");
		exit(1);
	}
	ch = getc(pgmFile);
	/*μετατροπή από char σε int του χαρακτήρα που δηλώνει τον τύπο της εικόνας*/
	type = ch - 48;
	if (type != 2)
	{
		printf("ERROR(2): Not valid pgm file type. Currently only P2 files can be used.\n");
		exit(1);
	}

	while (getc(pgmFile) != '\n');             // skip to end of line
	fseek(pgmFile, -1, SEEK_CUR);             // backup one character

	fscanf(pgmFile, "%d", &((*pgm).width));     //διάβασμα και αποθήκευση της τιμής του πλάτους της εικόνας
	fscanf(pgmFile, "%d", &((*pgm).height));    //διάβασμα και αποθήκευση της τιμής του ύψους της εικόνας
	fscanf(pgmFile, "%d", &((*pgm).maxGrey));   //διάβασμα και αποθήκευση της τιμής του μέγιστου επιπέδου γκρι της εικόνας

	printf("\n width  = %d", (*pgm).width);
	printf("\n height = %d", (*pgm).height);
	printf("\n maxVal = %d", (*pgm).maxGrey);
	printf("\n");

	if (type == 2) 
	{
		/*Δημιουργία του δυναμικού 2-διάστατου πίνακα "matrix". Στην ουσία δημιουργούμε πρώτα έναν μονοδιάστατο πίνακα,
		μήκους ίσο με το ύψος της εικόνας, ο οποίος θα περιέχει pointers προς άλλους μονοδιάστατους πίνακες, 
		μήκους ίσο με το πλάτος της εικόνας ο καθένας, στους οποίους τελικά θα αποθηκεύσουμε τις τιμές του γκρι κάθε pixel.*/
		
		pgm->matrix = (int **)malloc(sizeof(int *) * pgm->height);   //Κατανομή μνήμης για τον πίνακα με τα pointers. 
		if (pgm->matrix == NULL) {                                   //Σε περίπτωση σφάλματος κατά την κατανομή μνήμης εμφανίζουμε κατάλληλο μήνυμα.
			perror("memory allocation failure");
			exit(EXIT_FAILURE);
		}

		for (int i = 0; i < pgm->height; i++) {
			(*pgm).matrix[i] = (int *)malloc(sizeof(int) * (*pgm).width); //Κατανομή μνήμης για κάθε πίνακα που θα αποθηκεύει τις τιμές των pixel 
			if ((*pgm).matrix[i] == NULL) {                               //και πέρασμα του pointer προς αυτον στον προηγούμενο πίνακα.
				perror("memory allocation failure"); 					  //Σε περίπτωση σφάλματος κατά την κατανομή μνήμης εμφανίζουμε κατάλληλο μήνυμα.
				exit(EXIT_FAILURE);
			}
		}

		for (row = 0; row <pgm->height; row++)
			for (col = 0; col< pgm->width; col++)
			{
				fscanf(pgmFile, "%d", &ch_int);						//Διάβασμα της τιμής του γκρι για κάθε πίξελ και αποθήκευση της στο αντίστοιχο 
				(*pgm).matrix[row][col] = ch_int;					//στοιχείο του δισδιάστατου πίνακα "matrix".
			}
	}
	fclose(pgmFile);
	return type;
}



void cpuSharpenImg3x3(PGMImage *pgm)					//Συνάρτηση για την εφαρμογή του φίλτρου 3x3 χρησιμοποιώντας την CPU.
{
	
	int adj = 0;
	int i, j, k, m;

	/*Δημιουργούμε και πάλι έναν δισδιάστατο πίνακα, τον "matrixTr", στον οποίο θα αποθηκεύσουμε 
	τις τροποποιημένες σύμφωνα με το φίλτρο τιμές του πίνακα "matrix".*/
	pgm->matrixTr = (int **)malloc(sizeof(int *) * pgm->height);
	if (pgm->matrixTr == NULL) {
		perror("memory allocation failure");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < pgm->height; i++) {
		pgm->matrixTr[i] = (int *)malloc(sizeof(int) * pgm->width);
		if (pgm->matrixTr[i] == NULL) {
			perror("memory allocation failure");
			exit(EXIT_FAILURE);
		}
	}
	
	//Προσπέλαση όλων των στοιχείων του πίνακα "matrix", εκτός από αυτά που βρίσκονται στις πρώτες και τελευταίες γραμμές και στήλες, 
	//εφαρμογή του φίλτρου σε αυτά και αποθήκευση τους στον πίνακα "matrixTr".
	for (i = 1; i < pgm->height - 1; i++)
	{
		for (j = 1; j < pgm->width - 1; j++)
		{
			for (k = i - 1; k < i + 2; k++)
			{
				for (m = j - 1; m < j + 2; m++)
				{
					if (!((k == i) && (m == j)))
					{
						adj += pgm->matrix[k][m] * (-1);		
					}
				}
			}
			if ((pgm->matrix[i][j] * 9) + adj > 255)
			{
				pgm->matrixTr[i][j] = 255;
			}
			else if ((pgm->matrix[i][j] * 9) + adj <0)
			{
				pgm->matrixTr[i][j] = 0;
			}
			else
			{
				pgm->matrixTr[i][j] = ((pgm->matrix[i][j] * 9) + adj); 
			}
			adj = 0;
		}
	}
}

void cpuSharpenImg5x5(PGMImage *pgm)					//Συνάρτηση για την εφαρμογή του φίλτρου 5x5 χρησιμοποιώντας την CPU.
{
	int adj = 0;
	int i, j, k, m;
	/*Δημιουργούμε και πάλι έναν δισδιάστατο πίνακα, τον "matrixTr", στον οποίο θα αποθηκεύσουμε 
	τις τροποποιημένες σύμφωνα με το φίλτρο τιμές του πίνακα "matrix".*/
	pgm->matrixTr = (int **)malloc(sizeof(int *) * pgm->height);
	if (pgm->matrixTr == NULL) {
		perror("memory allocation failure");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < pgm->height; i++) {
		pgm->matrixTr[i] = (int *)malloc(sizeof(int) * pgm->width);
		if (pgm->matrixTr[i] == NULL) {
			perror("memory allocation failure");
			exit(EXIT_FAILURE);
		}
	}

	//Προσπέλαση όλων των στοιχείων του πίνακα "matrix", εκτός από αυτά που βρίσκονται στις 2 πρώτες και 2 τελευταίες γραμμές και στήλες, 
	//εφαρμογή του φίλτρου σε αυτά και αποθήκευση τους στον πίνακα "matrixTr".
	for (i = 2; i < pgm->height - 2; i++)
	{
		for (j = 2; j < pgm->width - 2; j++)
		{
			for (k = i - 2; k < i + 3; k++)
			{
				for (m = j - 2; m < j + 3; m++)
				{
					if (!((k == i) && (m == j)))
					{
						if ((k == i - 2) || (k == i + 2) || (m == j - 2) || (m == j + 2))
						{
							adj += pgm->matrix[k][m] * (-1);
						}
						else
						{
							adj += pgm->matrix[k][m] * 2;
						}
					}

				}
			}
			if (((pgm->matrix[i][j] * 8) + adj) / 8 > 255)
			{
				pgm->matrixTr[i][j] = 255;
			}
			else if (((pgm->matrix[i][j] * 8) + adj) / 8 <0)
			{
				pgm->matrixTr[i][j] = 0;
			}
			else
			{
				pgm->matrixTr[i][j] = ((pgm->matrix[i][j] * 8) + adj) / 8;
			}
			adj = 0;
		}
	}
}

void crNewFile3x3(const char *flnm, PGMImage *pgm, int type)			//Συνάρτηση για την δημιουργία καινούριων αρχείων .pgm, τα οποία περιέχουν την φιλτραρισμένη πλέον, σύμφωνα με το φίλτρο 3x3, εικόνα.
{
	FILE *pgm3x3File;
	char *ending = "3x3.pgm\0";			//Χαρακτήρες που θα μπουν στο τέλος του ονόματος του νέου αρχείου.
	size_t len = strlen(flnm);
	size_t len2 = strlen(ending);
	char *finalName = (char*) malloc(len + len2);
	strcpy(finalName, flnm);			 //Δημιουργία του ονόματος του νέου αρχείου με τον συνδυασμό του ονόματος του αρχικού αρχείου
	strcat(finalName, ending);			 //και τους χαρακτήρες "3x3.pgm\0".
	int  i, j;

	pgm3x3File = fopen(finalName, "w");		//Δημιουργία/άνοιγμα του αρχείου για εγγραφή.
	
	//Αποθήκευση των πληροφοριών της εικόνας(τύπος, πλάτος, ύψος και μέγιστη τιμή γκρι) στο αρχείο.
	fprintf(pgm3x3File, "%c", 'P');
	fprintf(pgm3x3File, "%d\n", 2);
	fprintf(pgm3x3File, "%d ", pgm->width);
	fprintf(pgm3x3File, "%d\n", pgm->height);
	fprintf(pgm3x3File, "%d\n", pgm->maxGrey);
	
	//Αποθήκευση των τιμών του γκρί κάθε πίξελ στο αρχείο.
	for (i = 0; i < pgm->height; i++)
	{
		for (j = 0; j < pgm->width; j++)
		{
			if ((i == 0) || (i == pgm->height - 1) || (j == 0) || (j == pgm->width - 1))
			{
				fprintf(pgm3x3File, "%d  ", pgm->matrix[i][j]);			//Διάβασμα των τιμών από τον πίνακα "matrix" για τα στοιχεία που δεν έχουμε εφαρμόσει το φίλτρο.

				if ((!((j == 0) && (i == 0))) && (j % 10 == 0))
				{
					fprintf(pgm3x3File, "%c\n", ' ');
				}
			}
			else
			{
				fprintf(pgm3x3File, "%d  ", pgm->matrixTr[i][j]);		//Διάβασμα των τιμών από τον πίνακα "matrixTr" για όλα τα υπόλοιπα στοιχεία.
				if (j % 10 == 0)
				{
					fprintf(pgm3x3File, "%c\n", ' ');
				}
			}
		}
	}
	fclose(pgm3x3File);
}

void crNewFile5x5(const char *flnm, PGMImage *pgm, int type)		//Συνάρτηση αντίστοιχη της "crNewFile3x3", απλά για εικόνες στις οποίες έχει εφαρμοστεί το φίλτρο 5x5.
{
	FILE *pgm5x5File;
	char *ending = "5x5.pgm\0";
	size_t len = strlen(flnm);
	size_t len2 = strlen(ending);
	char *finalName = (char*)malloc(len + len2);
	strcpy(finalName, flnm);
	strcat(finalName, ending);
	int  i, j;

	pgm5x5File = fopen(finalName, "w");
	fprintf(pgm5x5File, "%c", 'P');
	fprintf(pgm5x5File, "%d\n", 2);
	fprintf(pgm5x5File, "%d ", pgm->width);
	fprintf(pgm5x5File, "%d\n", pgm->height);
	fprintf(pgm5x5File, "%d\n", pgm->maxGrey);
	for (i = 0; i < pgm->height; i++)
	{
		for (j = 0; j < pgm->width; j++)
		{
			if ((i <2) || (i > pgm->height - 3) || (j <2) || (j > pgm->width - 3))
			{
				fprintf(pgm5x5File, "%d  ", pgm->matrix[i][j]);
				if ((j>0) && (j % 10 == 0))
				{
					fprintf(pgm5x5File, "%c\n", ' ');
				}
			}
			else
			{
				fprintf(pgm5x5File, "%d  ", pgm->matrixTr[i][j]);
				if (j % 10 == 0)
				{
					fprintf(pgm5x5File, "%c\n", ' ');
				}
			}
		}
	}
	fclose(pgm5x5File);
}

void createMTr(PGMImage *pgm)						//Συνάρτηση για την κατανομή μνήμης για τον πίνακα "matrixTr".
{
	pgm->matrixTr = (int **)malloc(sizeof(int *) * pgm->height);
	if (pgm->matrixTr == NULL) {
		perror("memory allocation failure");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < pgm->height; i++) {
		pgm->matrixTr[i] = (int *)malloc(sizeof(int) * pgm->width);
		if (pgm->matrixTr[i] == NULL) {
			perror("memory allocation failure");
			exit(EXIT_FAILURE);
		}
	}
}

void populateHMatrix(PGMImage *pgm,int* h_matrix)			//Συνάρτηση για την αντιγραφή κάθε στοιχείου του δισδιάστατου πίνακα "matrix" στον μονοδιάστατο πίνακα "h_matrix".
{
	int i, j;
	for (i = 0; i < pgm->height; i++)
	{
		for (j = 0; j < pgm->width; j++)
		{
			h_matrix[i*(pgm->height)+j] = pgm->matrix[i][j];
		}
	}
	
}

int main()
{
	clock_t t;
	clock_t startT;
	double dt;
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int type;
	char choice;
	int * h_matrix; 
	int *  d_matrix;
	int * d_matrixTr;
	int m = 64; //αριθμός threads.
	int n; //αριθμός blocks.
	PGMImage img;
	FILE *logFile;
	startT = clock();
	t = startT;
	logFile = fopen("sharpenlog.txt", "w");//Δημιουργία αρχείου για την καταγραφή του χρόνου που χρειάστηκε το πρόγραμμα για να ολοκληρώσει διάφορες εργασίες.
	printf("Please write the name of the file: \n");
	char filename[30];
	gets(filename);
	t = clock() - t;
	dt = ((double)t) / CLOCKS_PER_SEC;
	fprintf(logFile, "Filename was given after %f seconds.\n", dt);
	t = clock();
	type = getPGM(filename, &img);
	t = clock() - t;
	dt = ((double)t) / CLOCKS_PER_SEC;
	fprintf(logFile, "It took %f seconds to read the file.\n", dt);
	
	printf("Choose one of the following options by typing the appropriate number and pressing 'Enter': \n");
	printf("1. Apply 3x3 filter using the CPU.\n");
	printf("2. Apply 5x5 filter using the CPU.\n");
	printf("3. Apply 3x3 filter using the GPU.\n");
	printf("4. Apply 5x5 filter using the GPU.\n");
	choice = getchar();
	if (choice == '1')
	{
		t = clock();
		cpuSharpenImg3x3(&img); //Εφαρμογή του φίλτρου 3x3 με την χρήση CPU.
		t = clock() - t;
		dt = ((double)t) / CLOCKS_PER_SEC;
		fprintf(logFile, "It took %f seconds to apply the 3x3 filter.\n", dt);
		crNewFile3x3(filename, &img, type); //Δημιουργία του νέου αρχείου .pgm.
		deallocate_dynamic_matrix(img.matrixTr, img.height);
	}
	else if (choice == '2')
	{
		t = clock();
		cpuSharpenImg5x5(&img);  //Εφαρμογή του φίλτρου 5x5 με την χρήση CPU.
		t = clock() - t;
		dt = ((double)t) / CLOCKS_PER_SEC;
		fprintf(logFile, "It took %f seconds to apply the 5x5 filter.\n", dt);
		crNewFile5x5(filename, &img, type);  //Δημιουργία του νέου αρχείου .pgm.
		deallocate_dynamic_matrix(img.matrixTr, img.height);
	}
	else if (choice == '3')
	{
		n = (img.height*img.width) / m;//Καθορισμός του αριθμού των blocks.
		h_matrix = (int*)malloc(sizeof(int)*img.width*img.height); //Κατανομή μνήμης για τον πίνακα "h_matrix".
		if (h_matrix == NULL) 
		{
			perror("memory allocation failure");
			exit(EXIT_FAILURE);
		}
		populateHMatrix(&img, h_matrix); //Αντιγραφή των στοιχείων του "matrix" στον "h_matrix".
		createMTr(&img);//Κατανομή μνήμης για τον πίνακα "matrixTr".
		t = clock();
		cudaMalloc(&d_matrix, img.height*img.width*sizeof(int));//Κατανομή μνήμης στην GPU για τον πίνακα "d_matrix".
		cudaMalloc(&d_matrixTr, img.height*img.width*sizeof(int));//Κατανομή μνήμης στην GPU για τον πίνακα "d_matrixTr".
		cudaMemcpy(d_matrix, h_matrix, img.height*img.width*sizeof(int), cudaMemcpyHostToDevice);//Αντιγραφή του "h_matrix" στον "d_matrix".
		cudaDeviceSynchronize();
		t = clock() - t;
		dt = ((double)t) / CLOCKS_PER_SEC;
		fprintf(logFile, "It took %f seconds to allocate memory on the gpu and copy the data.\n", dt);
		
		cudaEventRecord(start, 0);
		gpuSharpenImg3x3<<<n,m>>>(d_matrix,d_matrixTr, img.width, img.height);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaError_t errAsync = cudaDeviceSynchronize();
		cudaEventElapsedTime(&time, start, stop);
		fprintf(logFile, "It took %f ms to apply the 3x3 filter using the GPU.\n", time);
		cudaMemcpy(h_matrix, d_matrixTr, img.height*img.width*sizeof(int), cudaMemcpyDeviceToHost);//Αντιγραφή του "d_matrixTr" στον "h_matrix".
		cudaDeviceSynchronize();
		for (int i = 0; i < img.height; i++)
		{
			for (int j = 0; j < img.width; j++)
			{
				img.matrixTr[i][j] = h_matrix[i*(img.height/* - 1*/) + j];//Αντιγραφή των στοιχείων του "h_matrix" στον "matrixTr".
			}
		}  
		crNewFile3x3(filename, &img, type);	   //Κάλεσμα της συνάρτησης για την δημιουργία του νέου αρχείου.
		
		//Διαγραφή πινάκων και απελευθέρωση της μνήμης που είχε δεσμευτεί για αυτούς.
		cudaFree(d_matrix);
		cudaFree(d_matrixTr);
		cudaDeviceSynchronize();
		deallocate_dynamic_matrix(img.matrixTr, img.height);
		free(h_matrix);			  
	}
	else if (choice == '4')
	{
		n = (img.height*img.width) / m;//Καθορισμός του αριθμού των blocks.
		h_matrix = (int*)malloc(sizeof(int)*img.width*img.height); //Κατανομή μνήμης για τον πίνακα "h_matrix".
		if (h_matrix == NULL) {
			perror("memory allocation failure");
			exit(EXIT_FAILURE);
		}
		populateHMatrix(&img, h_matrix);  //Αντιγραφή των στοιχείων του "matrix" στον "h_matrix".
		createMTr(&img);  //Κατανομή μνήμης για τον πίνακα "matrixTr".
		t = clock();
		cudaMalloc(&d_matrix, img.height*img.width*sizeof(int));//Κατανομή μνήμης στην GPU για τον πίνακα "d_matrix".
		cudaMalloc(&d_matrixTr, img.height*img.width*sizeof(int));//Κατανομή μνήμης στην GPU για τον πίνακα "d_matrixTr".
		cudaMemcpy(d_matrix, h_matrix, img.height*img.width*sizeof(int), cudaMemcpyHostToDevice);//Αντιγραφή του "h_matrix" στον "d_matrix".
		cudaDeviceSynchronize();
		t = clock() - t;
		dt = ((double)t) / CLOCKS_PER_SEC;
		fprintf(logFile, "It took %f seconds to allocate memory on the gpu and copy the data.\n", dt);
		cudaEventRecord(start, 0);
		gpuSharpenImg5x5<<<n,m>>>(d_matrix,d_matrixTr, img.width, img.height);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaError_t errAsync = cudaDeviceSynchronize();
		
		cudaEventElapsedTime(&time, start, stop);
		fprintf(logFile, "It took %f ms to apply the 5x5 filter using the GPU.\n", time);
		cudaMemcpy(h_matrix, d_matrixTr, img.height*img.width*sizeof(int), cudaMemcpyDeviceToHost);//Αντιγραφή του "d_matrixTr" στον "h_matrix".
		cudaDeviceSynchronize();
		for (int i = 0; i < img.height; i++)
		{
			for (int j = 0; j < img.width; j++)
			{
				img.matrixTr[i][j] = h_matrix[i*(img.height/* - 1*/) + j];//Αντιγραφή των στοιχείων του "h_matrix" στον "matrixTr".
			}
		}  
		crNewFile5x5(filename, &img, type);		   //Κάλεσμα της συνάρτησης για την δημιουργία του νέου αρχείου.

		//Διαγραφή πινάκων και απελευθέρωση της μνήμης που είχε δεσμευτεί για αυτούς.
		cudaFree(d_matrix);
		cudaFree(d_matrixTr);
		cudaDeviceSynchronize();
		deallocate_dynamic_matrix(img.matrixTr, img.height);
		free(h_matrix);
	}
	else
	{
		printf("Invalid option.");
		deallocate_dynamic_matrix(img.matrix, img.height);
		exit(1);
	}
	deallocate_dynamic_matrix(img.matrix, img.height);
	t = clock() - startT;
	dt = ((double)t) / CLOCKS_PER_SEC;
	fprintf(logFile, "It took %f seconds for the program to complete.\n", dt);
	fclose(logFile);
	return 0;

}