/* NOTE - eventually this file will be automatically updated using a Perl script that understand
 * the naming of test case files, functions, and namespaces.
 */

#include <time.h>   /* for time() */
#include <stdlib.h> /* for srand() */

#include "std_testcase.h"
#include "testcases.h"

int main(int argc, char * argv[]) {

	/* seed randomness */

	srand( (unsigned)time(NULL) );

	global_argc = argc;
	global_argv = argv;

#ifndef OMITGOOD

	/* Calling C good functions */
	/* BEGIN-AUTOGENERATED-C-GOOD-FUNCTION-CALLS */
	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_01_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_01_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_02_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_02_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_03_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_03_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_04_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_04_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_05_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_05_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_06_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_06_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_07_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_07_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_08_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_08_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_09_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_09_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_10_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_10_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_11_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_11_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_12_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_12_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_13_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_13_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_14_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_14_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_15_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_15_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_16_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_16_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_17_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_17_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_18_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_18_good();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_19_good();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_19_good();

	/* END-AUTOGENERATED-C-GOOD-FUNCTION-CALLS */





#ifdef __cplusplus
	/* Calling C++ good functions */
	/* BEGIN-AUTOGENERATED-CPP-GOOD-FUNCTION-CALLS */

	/* END-AUTOGENERATED-CPP-GOOD-FUNCTION-CALLS */

#endif /* __cplusplus */

#endif /* OMITGOOD */

#ifndef OMITBAD

	/* Calling C bad functions */
	/* BEGIN-AUTOGENERATED-C-BAD-FUNCTION-CALLS */
	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_01_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_01_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_02_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_02_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_03_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_03_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_04_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_04_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_05_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_05_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_06_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_06_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_07_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_07_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_08_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_08_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_09_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_09_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_10_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_10_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_11_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_11_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_12_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_12_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_13_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_13_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_14_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_14_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_15_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_15_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_16_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_16_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_17_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_17_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_18_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_18_bad();

	printLine("Calling CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_19_bad();");
	CWE587_Assignment_Of_Fixed_Address_To_Pointer__basic_19_bad();

	/* END-AUTOGENERATED-C-BAD-FUNCTION-CALLS */




	
#ifdef __cplusplus
	/* Calling C++ bad functions */
	/* BEGIN-AUTOGENERATED-CPP-BAD-FUNCTION-CALLS */

	/* END-AUTOGENERATED-CPP-BAD-FUNCTION-CALLS */
	
#endif /* __cplusplus */

#endif /* OMITBAD */

	return 0;

} 
