/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE23_Relative_Path_Traversal__wchar_t_Environment_w32CreateFile_53a.c
Label Definition File: CWE23_Relative_Path_Traversal.label.xml
Template File: sources-sink-53a.tmpl.c
*/
/*
 * @description
 * CWE: 23 Relative Path Traversal
 * BadSource: Environment Read input from an environment variable
 * GoodSource: File name without a period or slash
 * Sink: w32CreateFile
 *    BadSink :
 * Flow Variant: 53 Data flow: data passed as an argument from one function through two others to a fourth; all four functions are in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
#define BASEPATH L"c:\\temp\\"
#else
#define BASEPATH L"/tmp/"
#endif

#define ENV_VARIABLE L"ADD"

#ifdef _WIN32
# define GETENV _wgetenv
#else
# define GETENV wgetenv
#endif

#include <windows.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE23_Relative_Path_Traversal__wchar_t_Environment_w32CreateFile_53b_bad_sink(wchar_t * data);

void CWE23_Relative_Path_Traversal__wchar_t_Environment_w32CreateFile_53_bad()
{
    wchar_t * data;
    wchar_t data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    {
        /* Read input from an environment variable */
        size_t data_len = wcslen(data);
        wchar_t * environment = GETENV(ENV_VARIABLE);
        /* If there is data in the environment variable */
        if (environment != NULL)
        {
            wcsncat(data+data_len, environment, 100-data_len-1);
        }
    }
    CWE23_Relative_Path_Traversal__wchar_t_Environment_w32CreateFile_53b_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE23_Relative_Path_Traversal__wchar_t_Environment_w32CreateFile_53b_goodG2B_sink(wchar_t * data);

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    wchar_t * data;
    wchar_t data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    /* FIX: File name does not contain a period or slash */
    wcscat(data, L"file.txt");
    CWE23_Relative_Path_Traversal__wchar_t_Environment_w32CreateFile_53b_goodG2B_sink(data);
}

void CWE23_Relative_Path_Traversal__wchar_t_Environment_w32CreateFile_53_good()
{
    goodG2B();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE23_Relative_Path_Traversal__wchar_t_Environment_w32CreateFile_53_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE23_Relative_Path_Traversal__wchar_t_Environment_w32CreateFile_53_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
