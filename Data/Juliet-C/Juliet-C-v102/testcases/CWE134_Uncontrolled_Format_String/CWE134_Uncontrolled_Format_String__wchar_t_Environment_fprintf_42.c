/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__wchar_t_Environment_fprintf_42.c
Label Definition File: CWE134_Uncontrolled_Format_String.label.xml
Template File: sources-sinks-42.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Copy a fixed string into data
 * Sinks: fprintf
 *    GoodSink: fwprintf with "%s" as the second argument and data as the third
 *    BadSink : fwprintf with data as the second argument
 * Flow Variant: 42 Data flow: data returned from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#define ENV_VARIABLE L"ADD"

#ifdef _WIN32
# define GETENV _wgetenv
#else
# define GETENV wgetenv
#endif

#ifndef OMITBAD

static wchar_t * bad_source(wchar_t * data)
{
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
    return data;
}

void CWE134_Uncontrolled_Format_String__wchar_t_Environment_fprintf_42_bad()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    data = bad_source(data);
    /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
    fwprintf(stdout, data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
static wchar_t * goodG2B_source(wchar_t * data)
{
    /* FIX: Use a fixed string that does not contain a format specifier */
    wcscpy(data, L"fixedstringtest");
    return data;
}

static void goodG2B()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    data = goodG2B_source(data);
    /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
    fwprintf(stdout, data);
}

/* goodB2G uses the BadSource with the GoodSink */
static wchar_t * goodB2G_source(wchar_t * data)
{
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
    return data;
}

static void goodB2G()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    data = goodB2G_source(data);
    /* FIX: Specify the format disallowing a format string vulnerability */
    fwprintf(stdout, L"%s\n", data);
}

void CWE134_Uncontrolled_Format_String__wchar_t_Environment_fprintf_42_good()
{
    goodB2G();
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
    CWE134_Uncontrolled_Format_String__wchar_t_Environment_fprintf_42_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE134_Uncontrolled_Format_String__wchar_t_Environment_fprintf_42_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
