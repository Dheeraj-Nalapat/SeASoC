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
	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_01_good();");
	CWE123_Write_What_Where_Condition__connect_socket_01_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_02_good();");
	CWE123_Write_What_Where_Condition__connect_socket_02_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_03_good();");
	CWE123_Write_What_Where_Condition__connect_socket_03_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_04_good();");
	CWE123_Write_What_Where_Condition__connect_socket_04_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_05_good();");
	CWE123_Write_What_Where_Condition__connect_socket_05_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_06_good();");
	CWE123_Write_What_Where_Condition__connect_socket_06_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_07_good();");
	CWE123_Write_What_Where_Condition__connect_socket_07_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_08_good();");
	CWE123_Write_What_Where_Condition__connect_socket_08_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_09_good();");
	CWE123_Write_What_Where_Condition__connect_socket_09_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_10_good();");
	CWE123_Write_What_Where_Condition__connect_socket_10_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_11_good();");
	CWE123_Write_What_Where_Condition__connect_socket_11_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_12_good();");
	CWE123_Write_What_Where_Condition__connect_socket_12_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_13_good();");
	CWE123_Write_What_Where_Condition__connect_socket_13_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_14_good();");
	CWE123_Write_What_Where_Condition__connect_socket_14_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_15_good();");
	CWE123_Write_What_Where_Condition__connect_socket_15_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_16_good();");
	CWE123_Write_What_Where_Condition__connect_socket_16_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_17_good();");
	CWE123_Write_What_Where_Condition__connect_socket_17_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_18_good();");
	CWE123_Write_What_Where_Condition__connect_socket_18_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_19_good();");
	CWE123_Write_What_Where_Condition__connect_socket_19_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_31_good();");
	CWE123_Write_What_Where_Condition__connect_socket_31_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_32_good();");
	CWE123_Write_What_Where_Condition__connect_socket_32_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_34_good();");
	CWE123_Write_What_Where_Condition__connect_socket_34_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_41_good();");
	CWE123_Write_What_Where_Condition__connect_socket_41_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_44_good();");
	CWE123_Write_What_Where_Condition__connect_socket_44_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_45_good();");
	CWE123_Write_What_Where_Condition__connect_socket_45_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_51_good();");
	CWE123_Write_What_Where_Condition__connect_socket_51_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_52_good();");
	CWE123_Write_What_Where_Condition__connect_socket_52_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_53_good();");
	CWE123_Write_What_Where_Condition__connect_socket_53_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_54_good();");
	CWE123_Write_What_Where_Condition__connect_socket_54_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_63_good();");
	CWE123_Write_What_Where_Condition__connect_socket_63_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_64_good();");
	CWE123_Write_What_Where_Condition__connect_socket_64_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_65_good();");
	CWE123_Write_What_Where_Condition__connect_socket_65_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_66_good();");
	CWE123_Write_What_Where_Condition__connect_socket_66_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_67_good();");
	CWE123_Write_What_Where_Condition__connect_socket_67_good();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_68_good();");
	CWE123_Write_What_Where_Condition__connect_socket_68_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_01_good();");
	CWE123_Write_What_Where_Condition__fgets_01_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_02_good();");
	CWE123_Write_What_Where_Condition__fgets_02_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_03_good();");
	CWE123_Write_What_Where_Condition__fgets_03_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_04_good();");
	CWE123_Write_What_Where_Condition__fgets_04_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_05_good();");
	CWE123_Write_What_Where_Condition__fgets_05_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_06_good();");
	CWE123_Write_What_Where_Condition__fgets_06_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_07_good();");
	CWE123_Write_What_Where_Condition__fgets_07_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_08_good();");
	CWE123_Write_What_Where_Condition__fgets_08_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_09_good();");
	CWE123_Write_What_Where_Condition__fgets_09_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_10_good();");
	CWE123_Write_What_Where_Condition__fgets_10_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_11_good();");
	CWE123_Write_What_Where_Condition__fgets_11_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_12_good();");
	CWE123_Write_What_Where_Condition__fgets_12_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_13_good();");
	CWE123_Write_What_Where_Condition__fgets_13_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_14_good();");
	CWE123_Write_What_Where_Condition__fgets_14_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_15_good();");
	CWE123_Write_What_Where_Condition__fgets_15_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_16_good();");
	CWE123_Write_What_Where_Condition__fgets_16_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_17_good();");
	CWE123_Write_What_Where_Condition__fgets_17_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_18_good();");
	CWE123_Write_What_Where_Condition__fgets_18_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_19_good();");
	CWE123_Write_What_Where_Condition__fgets_19_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_31_good();");
	CWE123_Write_What_Where_Condition__fgets_31_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_32_good();");
	CWE123_Write_What_Where_Condition__fgets_32_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_34_good();");
	CWE123_Write_What_Where_Condition__fgets_34_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_41_good();");
	CWE123_Write_What_Where_Condition__fgets_41_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_44_good();");
	CWE123_Write_What_Where_Condition__fgets_44_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_45_good();");
	CWE123_Write_What_Where_Condition__fgets_45_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_51_good();");
	CWE123_Write_What_Where_Condition__fgets_51_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_52_good();");
	CWE123_Write_What_Where_Condition__fgets_52_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_53_good();");
	CWE123_Write_What_Where_Condition__fgets_53_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_54_good();");
	CWE123_Write_What_Where_Condition__fgets_54_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_63_good();");
	CWE123_Write_What_Where_Condition__fgets_63_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_64_good();");
	CWE123_Write_What_Where_Condition__fgets_64_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_65_good();");
	CWE123_Write_What_Where_Condition__fgets_65_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_66_good();");
	CWE123_Write_What_Where_Condition__fgets_66_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_67_good();");
	CWE123_Write_What_Where_Condition__fgets_67_good();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_68_good();");
	CWE123_Write_What_Where_Condition__fgets_68_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_01_good();");
	CWE123_Write_What_Where_Condition__listen_socket_01_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_02_good();");
	CWE123_Write_What_Where_Condition__listen_socket_02_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_03_good();");
	CWE123_Write_What_Where_Condition__listen_socket_03_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_04_good();");
	CWE123_Write_What_Where_Condition__listen_socket_04_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_05_good();");
	CWE123_Write_What_Where_Condition__listen_socket_05_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_06_good();");
	CWE123_Write_What_Where_Condition__listen_socket_06_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_07_good();");
	CWE123_Write_What_Where_Condition__listen_socket_07_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_08_good();");
	CWE123_Write_What_Where_Condition__listen_socket_08_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_09_good();");
	CWE123_Write_What_Where_Condition__listen_socket_09_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_10_good();");
	CWE123_Write_What_Where_Condition__listen_socket_10_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_11_good();");
	CWE123_Write_What_Where_Condition__listen_socket_11_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_12_good();");
	CWE123_Write_What_Where_Condition__listen_socket_12_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_13_good();");
	CWE123_Write_What_Where_Condition__listen_socket_13_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_14_good();");
	CWE123_Write_What_Where_Condition__listen_socket_14_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_15_good();");
	CWE123_Write_What_Where_Condition__listen_socket_15_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_16_good();");
	CWE123_Write_What_Where_Condition__listen_socket_16_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_17_good();");
	CWE123_Write_What_Where_Condition__listen_socket_17_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_18_good();");
	CWE123_Write_What_Where_Condition__listen_socket_18_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_19_good();");
	CWE123_Write_What_Where_Condition__listen_socket_19_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_31_good();");
	CWE123_Write_What_Where_Condition__listen_socket_31_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_32_good();");
	CWE123_Write_What_Where_Condition__listen_socket_32_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_34_good();");
	CWE123_Write_What_Where_Condition__listen_socket_34_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_41_good();");
	CWE123_Write_What_Where_Condition__listen_socket_41_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_44_good();");
	CWE123_Write_What_Where_Condition__listen_socket_44_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_45_good();");
	CWE123_Write_What_Where_Condition__listen_socket_45_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_51_good();");
	CWE123_Write_What_Where_Condition__listen_socket_51_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_52_good();");
	CWE123_Write_What_Where_Condition__listen_socket_52_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_53_good();");
	CWE123_Write_What_Where_Condition__listen_socket_53_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_54_good();");
	CWE123_Write_What_Where_Condition__listen_socket_54_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_63_good();");
	CWE123_Write_What_Where_Condition__listen_socket_63_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_64_good();");
	CWE123_Write_What_Where_Condition__listen_socket_64_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_65_good();");
	CWE123_Write_What_Where_Condition__listen_socket_65_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_66_good();");
	CWE123_Write_What_Where_Condition__listen_socket_66_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_67_good();");
	CWE123_Write_What_Where_Condition__listen_socket_67_good();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_68_good();");
	CWE123_Write_What_Where_Condition__listen_socket_68_good();

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
	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_01_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_01_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_02_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_02_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_03_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_03_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_04_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_04_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_05_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_05_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_06_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_06_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_07_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_07_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_08_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_08_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_09_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_09_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_10_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_10_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_11_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_11_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_12_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_12_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_13_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_13_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_14_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_14_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_15_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_15_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_16_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_16_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_17_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_17_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_18_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_18_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_19_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_19_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_31_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_31_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_32_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_32_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_34_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_34_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_41_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_41_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_44_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_44_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_45_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_45_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_51_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_51_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_52_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_52_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_53_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_53_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_54_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_54_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_63_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_63_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_64_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_64_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_65_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_65_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_66_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_66_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_67_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_67_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__connect_socket_68_bad();");
	CWE123_Write_What_Where_Condition__connect_socket_68_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_01_bad();");
	CWE123_Write_What_Where_Condition__fgets_01_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_02_bad();");
	CWE123_Write_What_Where_Condition__fgets_02_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_03_bad();");
	CWE123_Write_What_Where_Condition__fgets_03_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_04_bad();");
	CWE123_Write_What_Where_Condition__fgets_04_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_05_bad();");
	CWE123_Write_What_Where_Condition__fgets_05_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_06_bad();");
	CWE123_Write_What_Where_Condition__fgets_06_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_07_bad();");
	CWE123_Write_What_Where_Condition__fgets_07_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_08_bad();");
	CWE123_Write_What_Where_Condition__fgets_08_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_09_bad();");
	CWE123_Write_What_Where_Condition__fgets_09_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_10_bad();");
	CWE123_Write_What_Where_Condition__fgets_10_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_11_bad();");
	CWE123_Write_What_Where_Condition__fgets_11_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_12_bad();");
	CWE123_Write_What_Where_Condition__fgets_12_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_13_bad();");
	CWE123_Write_What_Where_Condition__fgets_13_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_14_bad();");
	CWE123_Write_What_Where_Condition__fgets_14_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_15_bad();");
	CWE123_Write_What_Where_Condition__fgets_15_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_16_bad();");
	CWE123_Write_What_Where_Condition__fgets_16_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_17_bad();");
	CWE123_Write_What_Where_Condition__fgets_17_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_18_bad();");
	CWE123_Write_What_Where_Condition__fgets_18_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_19_bad();");
	CWE123_Write_What_Where_Condition__fgets_19_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_31_bad();");
	CWE123_Write_What_Where_Condition__fgets_31_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_32_bad();");
	CWE123_Write_What_Where_Condition__fgets_32_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_34_bad();");
	CWE123_Write_What_Where_Condition__fgets_34_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_41_bad();");
	CWE123_Write_What_Where_Condition__fgets_41_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_44_bad();");
	CWE123_Write_What_Where_Condition__fgets_44_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_45_bad();");
	CWE123_Write_What_Where_Condition__fgets_45_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_51_bad();");
	CWE123_Write_What_Where_Condition__fgets_51_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_52_bad();");
	CWE123_Write_What_Where_Condition__fgets_52_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_53_bad();");
	CWE123_Write_What_Where_Condition__fgets_53_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_54_bad();");
	CWE123_Write_What_Where_Condition__fgets_54_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_63_bad();");
	CWE123_Write_What_Where_Condition__fgets_63_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_64_bad();");
	CWE123_Write_What_Where_Condition__fgets_64_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_65_bad();");
	CWE123_Write_What_Where_Condition__fgets_65_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_66_bad();");
	CWE123_Write_What_Where_Condition__fgets_66_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_67_bad();");
	CWE123_Write_What_Where_Condition__fgets_67_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__fgets_68_bad();");
	CWE123_Write_What_Where_Condition__fgets_68_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_01_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_01_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_02_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_02_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_03_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_03_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_04_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_04_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_05_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_05_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_06_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_06_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_07_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_07_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_08_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_08_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_09_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_09_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_10_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_10_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_11_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_11_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_12_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_12_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_13_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_13_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_14_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_14_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_15_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_15_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_16_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_16_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_17_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_17_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_18_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_18_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_19_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_19_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_31_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_31_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_32_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_32_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_34_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_34_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_41_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_41_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_44_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_44_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_45_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_45_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_51_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_51_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_52_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_52_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_53_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_53_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_54_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_54_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_63_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_63_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_64_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_64_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_65_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_65_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_66_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_66_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_67_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_67_bad();

	printLine("Calling CWE123_Write_What_Where_Condition__listen_socket_68_bad();");
	CWE123_Write_What_Where_Condition__listen_socket_68_bad();

	/* END-AUTOGENERATED-C-BAD-FUNCTION-CALLS */




	
#ifdef __cplusplus
	/* Calling C++ bad functions */
	/* BEGIN-AUTOGENERATED-CPP-BAD-FUNCTION-CALLS */

	/* END-AUTOGENERATED-CPP-BAD-FUNCTION-CALLS */
	
#endif /* __cplusplus */

#endif /* OMITBAD */

	return 0;

} 
