/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE23_Relative_Path_Traversal__wchar_t_Environment_fopen_65a.c
Label Definition File: CWE23_Relative_Path_Traversal.label.xml
Template File: sources-sink-65a.tmpl.c
*/
/*
 * @description
 * CWE: 23 Relative Path Traversal
 * BadSource: Environment Read input from an environment variable
 * GoodSource: File name without a period or slash
 * Sinks: fopen
 *    BadSink :
 * Flow Variant: 65 Data/control flow: data passed as an argument from one function to a function in a different source file called via a function pointer
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

#ifdef _WIN32
# define FOPEN _wfopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifndef OMITBAD

/* bad function declaration */
void CWE23_Relative_Path_Traversal__wchar_t_Environment_fopen_65b_bad_sink(wchar_t * data);

void CWE23_Relative_Path_Traversal__wchar_t_Environment_fopen_65_bad()
{
    wchar_t * data;
    /* define a function pointer */
    void (*func_ptr) (wchar_t *) = CWE23_Relative_Path_Traversal__wchar_t_Environment_fopen_65b_bad_sink;
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
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE23_Relative_Path_Traversal__wchar_t_Environment_fopen_65b_goodG2B_sink(wchar_t * data);

static void goodG2B()
{
    wchar_t * data;
    void (*func_ptr) (wchar_t *) = CWE23_Relative_Path_Traversal__wchar_t_Environment_fopen_65b_goodG2B_sink;
    wchar_t data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    /* FIX: File name does not contain a period or slash */
    wcscat(data, L"file.txt");
    func_ptr(data);
}

void CWE23_Relative_Path_Traversal__wchar_t_Environment_fopen_65_good()
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
    CWE23_Relative_Path_Traversal__wchar_t_Environment_fopen_65_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE23_Relative_Path_Traversal__wchar_t_Environment_fopen_65_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
