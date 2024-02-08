/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE187_Partial_Comparison__wchar_t_substring_str_12.c
Label Definition File: CWE187_Partial_Comparison.label.xml
Template File: sources-sinks-12.tmpl.c
*/
/*
 * @description
 * CWE: 187 Partial Comparison
 * BadSource: substring Provide a password that is a shortened version of the actual password
 * GoodSource: Provide a matching password
 * Sinks: str
 *    GoodSink: Compare the 2 passwords correctly
 *    BadSink : use wcsstr() to do password match, which is a partial comparison
 * Flow Variant: 12 Control flow: if(global_returns_t_or_f())
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define PASSWORD L"Password1234"
/* PASSWORD_SZ must equal the length of PASSWORD */
#define PASSWORD_SZ wcslen(PASSWORD)

#ifndef OMITBAD

void CWE187_Partial_Comparison__wchar_t_substring_str_12_bad()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    if(global_returns_t_or_f())
    {
        /* FLAW: Provide a shortened version of the actual password
         * NOTE: This must be a substring of PASSWORD starting with the first character in PASSWORD
         * i.e. other examples could be "Pa", "Pas", "Pass", etc. as long as it is shorter than MIN_PASSWORD_SZ
         * and does not match the full PASSWORD string */
        data = L"P";
    }
    else
    {
        /* FIX: Use the matching string */
        data = PASSWORD;
    }
    if(global_returns_t_or_f())
    {
        /* POTENTIAL FLAW: Partially compare the two passwords */
        if (wcsstr(PASSWORD, data) != NULL)
            printLine("Access granted");
        else
            printLine("Access denied!");
    }
    else
    {
        /* Ideally, we would want to do a check to see if the passwords are of equal length */
        /* FIX: Compare the two passwords completely and correctly */
        if (wcscmp(PASSWORD, data) == 0)
            printLine("Access granted");
        else
            printLine("Access denied!");
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G() - use badsource and goodsink by changing the first "if" so that
   both branches use the BadSource and the second "if" so that both branches
   use the GoodSink */
static void goodB2G()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    if(global_returns_t_or_f())
    {
        /* FLAW: Provide a shortened version of the actual password
         * NOTE: This must be a substring of PASSWORD starting with the first character in PASSWORD
         * i.e. other examples could be "Pa", "Pas", "Pass", etc. as long as it is shorter than MIN_PASSWORD_SZ
         * and does not match the full PASSWORD string */
        data = L"P";
    }
    else
    {
        /* FLAW: Provide a shortened version of the actual password
         * NOTE: This must be a substring of PASSWORD starting with the first character in PASSWORD
         * i.e. other examples could be "Pa", "Pas", "Pass", etc. as long as it is shorter than MIN_PASSWORD_SZ
         * and does not match the full PASSWORD string */
        data = L"P";
    }
    if(global_returns_t_or_f())
    {
        /* Ideally, we would want to do a check to see if the passwords are of equal length */
        /* FIX: Compare the two passwords completely and correctly */
        if (wcscmp(PASSWORD, data) == 0)
            printLine("Access granted");
        else
            printLine("Access denied!");
    }
    else
    {
        /* Ideally, we would want to do a check to see if the passwords are of equal length */
        /* FIX: Compare the two passwords completely and correctly */
        if (wcscmp(PASSWORD, data) == 0)
            printLine("Access granted");
        else
            printLine("Access denied!");
    }
}

/* goodG2B() - use goodsource and badsink by changing the first "if" so that
   both branches use the GoodSource and the second "if" so that both branches
   use the BadSink */
static void goodG2B()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    if(global_returns_t_or_f())
    {
        /* FIX: Use the matching string */
        data = PASSWORD;
    }
    else
    {
        /* FIX: Use the matching string */
        data = PASSWORD;
    }
    if(global_returns_t_or_f())
    {
        /* POTENTIAL FLAW: Partially compare the two passwords */
        if (wcsstr(PASSWORD, data) != NULL)
            printLine("Access granted");
        else
            printLine("Access denied!");
    }
    else
    {
        /* POTENTIAL FLAW: Partially compare the two passwords */
        if (wcsstr(PASSWORD, data) != NULL)
            printLine("Access granted");
        else
            printLine("Access denied!");
    }
}

void CWE187_Partial_Comparison__wchar_t_substring_str_12_good()
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
    CWE187_Partial_Comparison__wchar_t_substring_str_12_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE187_Partial_Comparison__wchar_t_substring_str_12_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
