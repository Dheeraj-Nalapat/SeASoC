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
	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_01_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_01_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_02_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_02_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_03_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_03_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_04_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_04_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_05_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_05_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_06_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_06_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_07_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_07_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_08_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_08_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_09_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_09_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_10_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_10_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_11_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_11_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_12_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_12_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_13_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_13_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_14_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_14_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_15_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_15_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_16_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_16_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_17_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_17_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_18_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_18_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_19_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_19_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_01_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_01_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_02_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_02_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_03_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_03_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_04_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_04_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_05_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_05_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_06_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_06_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_07_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_07_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_08_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_08_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_09_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_09_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_10_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_10_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_11_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_11_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_12_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_12_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_13_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_13_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_14_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_14_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_15_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_15_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_16_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_16_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_17_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_17_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_18_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_18_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_19_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_19_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_01_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_01_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_02_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_02_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_03_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_03_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_04_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_04_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_05_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_05_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_06_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_06_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_07_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_07_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_08_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_08_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_09_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_09_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_10_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_10_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_11_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_11_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_12_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_12_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_13_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_13_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_14_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_14_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_15_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_15_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_16_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_16_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_17_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_17_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_18_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_18_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_19_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_19_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_01_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_01_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_02_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_02_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_03_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_03_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_04_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_04_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_05_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_05_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_06_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_06_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_07_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_07_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_08_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_08_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_09_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_09_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_10_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_10_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_11_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_11_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_12_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_12_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_13_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_13_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_14_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_14_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_15_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_15_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_16_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_16_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_17_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_17_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_18_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_18_good();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_19_good();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_19_good();

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
	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_01_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_01_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_02_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_02_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_03_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_03_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_04_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_04_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_05_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_05_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_06_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_06_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_07_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_07_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_08_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_08_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_09_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_09_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_10_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_10_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_11_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_11_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_12_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_12_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_13_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_13_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_14_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_14_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_15_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_15_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_16_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_16_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_17_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_17_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_18_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_18_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_19_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32alloca_19_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_01_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_01_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_02_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_02_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_03_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_03_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_04_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_04_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_05_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_05_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_06_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_06_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_07_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_07_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_08_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_08_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_09_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_09_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_10_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_10_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_11_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_11_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_12_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_12_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_13_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_13_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_14_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_14_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_15_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_15_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_16_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_16_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_17_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_17_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_18_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_18_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_19_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__char_w32declare_19_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_01_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_01_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_02_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_02_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_03_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_03_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_04_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_04_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_05_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_05_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_06_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_06_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_07_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_07_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_08_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_08_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_09_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_09_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_10_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_10_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_11_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_11_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_12_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_12_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_13_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_13_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_14_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_14_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_15_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_15_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_16_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_16_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_17_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_17_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_18_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_18_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_19_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32alloca_19_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_01_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_01_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_02_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_02_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_03_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_03_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_04_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_04_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_05_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_05_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_06_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_06_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_07_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_07_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_08_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_08_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_09_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_09_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_10_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_10_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_11_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_11_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_12_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_12_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_13_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_13_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_14_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_14_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_15_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_15_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_16_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_16_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_17_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_17_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_18_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_18_bad();

	printLine("Calling CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_19_bad();");
	CWE226_Sensitive_Information_Uncleared_Before_Release__wchar_t_w32declare_19_bad();

	/* END-AUTOGENERATED-C-BAD-FUNCTION-CALLS */




	
#ifdef __cplusplus
	/* Calling C++ bad functions */
	/* BEGIN-AUTOGENERATED-CPP-BAD-FUNCTION-CALLS */

	/* END-AUTOGENERATED-CPP-BAD-FUNCTION-CALLS */
	
#endif /* __cplusplus */

#endif /* OMITBAD */

	return 0;

} 
