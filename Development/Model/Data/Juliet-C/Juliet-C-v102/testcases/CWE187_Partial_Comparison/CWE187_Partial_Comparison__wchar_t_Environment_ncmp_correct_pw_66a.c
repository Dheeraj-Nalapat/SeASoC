/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66a.c
Label Definition File: CWE187_Partial_Comparison.label.xml
Template File: sources-sinks-66a.tmpl.c
*/
/*
 * @description
 * CWE: 187 Partial Comparison
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Provide a matching password
 * Sinks: ncmp_correct_pw
 *    GoodSink: Compare the 2 passwords correctly
 *    BadSink : use wcsncmp() to do password match, but use the length of the correct password
 * Flow Variant: 66 Data flow: data passed in an array from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define PASSWORD L"Password1234"
/* PASSWORD_SZ must equal the length of PASSWORD */
#define PASSWORD_SZ wcslen(PASSWORD)

#define ENV_VARIABLE L"ADD"

#ifdef _WIN32
# define GETENV _wgetenv
#else
# define GETENV wgetenv
#endif

#ifndef OMITBAD

/* bad function declaration */
void CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66b_bad_sink(wchar_t * data_array[]);

void CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66_bad()
{
    wchar_t * data;
    wchar_t * data_array[5];
    wchar_t data_buf[100] = L"";
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
    /* put data in array */
    data_array[2] = data;
    CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66b_bad_sink(data_array);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66b_goodG2B_sink(wchar_t * data_array[]);

static void goodG2B()
{
    wchar_t * data;
    wchar_t * data_array[5];
    wchar_t data_buf[100] = L"";
    data = data_buf;
    /* FIX: Use the matching string */
    data = PASSWORD;
    data_array[2] = data;
    CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66b_goodG2B_sink(data_array);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66b_goodB2G_sink(wchar_t * data_array[]);

static void goodB2G()
{
    wchar_t * data;
    wchar_t * data_array[5];
    wchar_t data_buf[100] = L"";
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
    data_array[2] = data;
    CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66b_goodB2G_sink(data_array);
}

void CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66_good()
{
    goodG2B();
    goodB2G();
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
    CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE187_Partial_Comparison__wchar_t_Environment_ncmp_correct_pw_66_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif