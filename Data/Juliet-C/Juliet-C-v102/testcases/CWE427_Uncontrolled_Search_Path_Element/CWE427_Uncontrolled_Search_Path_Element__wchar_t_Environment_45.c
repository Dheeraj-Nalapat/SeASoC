/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45.c
Label Definition File: CWE427_Uncontrolled_Search_Path_Element.label.xml
Template File: sources-sink-45.tmpl.c
*/
/*
 * @description
 * CWE: 427 Uncontrolled Search Path Element
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Use a hardcoded path
 * Sinks:
 *    BadSink : Set the environment variable
 * Flow Variant: 45 Data flow: data passed as a static global variable from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define NEW_PATH L"%SystemRoot%\\system32"
# define PUTENV _wputenv
#else
# define NEW_PATH L"/bin"
# define PUTENV wputenv
#endif

#define ENV_VARIABLE L"ADD"

#ifdef _WIN32
# define GETENV _wgetenv
#else
# define GETENV wgetenv
#endif

static wchar_t * CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45_bad_data;
static wchar_t * CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45_goodG2B_data;

#ifndef OMITBAD

static void bad_sink()
{
    wchar_t * data = CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45_bad_data;
    /* POTENTIAL FLAW: Set a new environment variable with a path that is possibly insecure */
    PUTENV(data);
}

void CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45_bad()
{
    wchar_t * data;
    wchar_t data_buf[250] = L"PATH=";
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
    CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45_bad_data = data;
    bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_sink()
{
    wchar_t * data = CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45_goodG2B_data;
    /* POTENTIAL FLAW: Set a new environment variable with a path that is possibly insecure */
    PUTENV(data);
}

static void goodG2B()
{
    wchar_t * data;
    wchar_t data_buf[250] = L"PATH=";
    data = data_buf;
    /* FIX: Set the path as the "system" path */
    wcscat(data, NEW_PATH);
    CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45_goodG2B_data = data;
    goodG2B_sink();
}

void CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45_good()
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
    CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE427_Uncontrolled_Search_Path_Element__wchar_t_Environment_45_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
