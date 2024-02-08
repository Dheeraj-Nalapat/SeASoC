/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE187_Partial_Comparison__wchar_t_fromFile_str_16.c
Label Definition File: CWE187_Partial_Comparison.label.xml
Template File: sources-sinks-16.tmpl.c
*/
/*
 * @description
 * CWE: 187 Partial Comparison
 * BadSource: fromFile Read input from a file
 * GoodSource: Provide a matching password
 * Sinks: str
 *    GoodSink: Compare the 2 passwords correctly
 *    BadSink : use wcsstr() to do password match, which is a partial comparison
 * Flow Variant: 16 Control flow: while(1) and while(0)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define PASSWORD L"Password1234"
/* PASSWORD_SZ must equal the length of PASSWORD */
#define PASSWORD_SZ wcslen(PASSWORD)

#ifdef _WIN32
# define FOPEN _wfopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifndef OMITBAD

void CWE187_Partial_Comparison__wchar_t_fromFile_str_16_bad()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    while(0)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use the matching string */
        data = PASSWORD;
        break;
    }
    while(1)
    {
        {
            /* Read input from a file */
            size_t data_len = wcslen(data);
            FILE * pFile;
            /* if there is room in data, attempt to read the input from a file */
            if(100-data_len > 1)
            {
                pFile = FOPEN(L"C:\\temp\\file.txt", L"r");
                if (pFile != NULL)
                {
                    fgetws(data+data_len, (int)(100-data_len), pFile);
                    fclose(pFile);
                }
            }
        }
        break;
    }
    while(1)
    {
        /* POTENTIAL FLAW: Partially compare the two passwords */
        if (wcsstr(PASSWORD, data) != NULL)
            printLine("Access granted");
        else
            printLine("Access denied!");
        break;
    }
    while(0)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* Ideally, we would want to do a check to see if the passwords are of equal length */
        /* FIX: Compare the two passwords completely and correctly */
        if (wcscmp(PASSWORD, data) == 0)
            printLine("Access granted");
        else
            printLine("Access denied!");
        break;
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G() - use badsource and goodsink by changing the conditions on the third and fourth while statements */
static void goodB2G()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    while(0)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use the matching string */
        data = PASSWORD;
        break;
    }
    while(1)
    {
        {
            /* Read input from a file */
            size_t data_len = wcslen(data);
            FILE * pFile;
            /* if there is room in data, attempt to read the input from a file */
            if(100-data_len > 1)
            {
                pFile = FOPEN(L"C:\\temp\\file.txt", L"r");
                if (pFile != NULL)
                {
                    fgetws(data+data_len, (int)(100-data_len), pFile);
                    fclose(pFile);
                }
            }
        }
        break;
    }
    while(0)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Partially compare the two passwords */
        if (wcsstr(PASSWORD, data) != NULL)
            printLine("Access granted");
        else
            printLine("Access denied!");
        break;
    }
    while(1)
    {
        /* Ideally, we would want to do a check to see if the passwords are of equal length */
        /* FIX: Compare the two passwords completely and correctly */
        if (wcscmp(PASSWORD, data) == 0)
            printLine("Access granted");
        else
            printLine("Access denied!");
        break;
    }
}

/* goodG2B() - use goodsource and badsink by changing the conditions on the first and second while statements */
static void goodG2B()
{
    wchar_t * data;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    while(1)
    {
        /* FIX: Use the matching string */
        data = PASSWORD;
        break;
    }
    while(0)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* Read input from a file */
            size_t data_len = wcslen(data);
            FILE * pFile;
            /* if there is room in data, attempt to read the input from a file */
            if(100-data_len > 1)
            {
                pFile = FOPEN(L"C:\\temp\\file.txt", L"r");
                if (pFile != NULL)
                {
                    fgetws(data+data_len, (int)(100-data_len), pFile);
                    fclose(pFile);
                }
            }
        }
        break;
    }
    while(1)
    {
        /* POTENTIAL FLAW: Partially compare the two passwords */
        if (wcsstr(PASSWORD, data) != NULL)
            printLine("Access granted");
        else
            printLine("Access denied!");
        break;
    }
    while(0)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* Ideally, we would want to do a check to see if the passwords are of equal length */
        /* FIX: Compare the two passwords completely and correctly */
        if (wcscmp(PASSWORD, data) == 0)
            printLine("Access granted");
        else
            printLine("Access denied!");
        break;
    }
}

void CWE187_Partial_Comparison__wchar_t_fromFile_str_16_good()
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
    CWE187_Partial_Comparison__wchar_t_fromFile_str_16_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE187_Partial_Comparison__wchar_t_fromFile_str_16_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
