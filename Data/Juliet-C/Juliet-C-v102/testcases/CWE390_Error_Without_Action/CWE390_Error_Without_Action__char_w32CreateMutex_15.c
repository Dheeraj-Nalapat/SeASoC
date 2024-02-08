/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE390_Error_Without_Action__char_w32CreateMutex_15.c
Label Definition File: CWE390_Error_Without_Action.string.label.xml
Template File: point-flaw-15.tmpl.c
*/
/*
 * @description
 * CWE: 390 Detection of Error Condition Without Action
 * Sinks: w32CreateMutex
 *    GoodSink: Check the return value of CreateMutexA() and handle it properly
 *    BadSink : Check to see if CreateMutexA() failed, but do nothing about it
 * Flow Variant: 15 Control flow: switch(6)
 *
 * */

#include "std_testcase.h"

#include <windows.h>
#define BUFSIZE 1024

#ifndef OMITBAD

void CWE390_Error_Without_Action__char_w32CreateMutex_15_bad()
{
    switch(6)
    {
    case 6:
    {
        HANDLE hMutex = NULL;
        hMutex = CreateMutexA(NULL, FALSE, NULL);
        /* FLAW: check for an error, but do nothing if one occurred */
        if (hMutex == NULL)
        {
            /* do nothing */
        }
        if (GetLastError() == ERROR_ALREADY_EXISTS)
        {
            /* do nothing */
        }
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hMutex);
    }
    break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        HANDLE hMutex = NULL;
        hMutex = CreateMutexA(NULL, FALSE, NULL);
        /* FIX: Check the return value of CreateMutex() for NULL AND
         * Check the return value of GetLastError() for ERROR_ALREADY_EXISTS */
        if (hMutex == NULL)
        {
            exit(1);
        }
        if (GetLastError() == ERROR_ALREADY_EXISTS)
        {
            exit(1);
        }
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hMutex);
    }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() changes the switch to switch(5) */
static void good1()
{
    switch(5)
    {
    case 6:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        HANDLE hMutex = NULL;
        hMutex = CreateMutexA(NULL, FALSE, NULL);
        /* FLAW: check for an error, but do nothing if one occurred */
        if (hMutex == NULL)
        {
            /* do nothing */
        }
        if (GetLastError() == ERROR_ALREADY_EXISTS)
        {
            /* do nothing */
        }
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hMutex);
    }
    break;
    default:
    {
        HANDLE hMutex = NULL;
        hMutex = CreateMutexA(NULL, FALSE, NULL);
        /* FIX: Check the return value of CreateMutex() for NULL AND
         * Check the return value of GetLastError() for ERROR_ALREADY_EXISTS */
        if (hMutex == NULL)
        {
            exit(1);
        }
        if (GetLastError() == ERROR_ALREADY_EXISTS)
        {
            exit(1);
        }
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hMutex);
    }
    }
}

/* good2() reverses the blocks in the switch */
static void good2()
{
    switch(6)
    {
    case 6:
    {
        HANDLE hMutex = NULL;
        hMutex = CreateMutexA(NULL, FALSE, NULL);
        /* FIX: Check the return value of CreateMutex() for NULL AND
         * Check the return value of GetLastError() for ERROR_ALREADY_EXISTS */
        if (hMutex == NULL)
        {
            exit(1);
        }
        if (GetLastError() == ERROR_ALREADY_EXISTS)
        {
            exit(1);
        }
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hMutex);
    }
    break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        HANDLE hMutex = NULL;
        hMutex = CreateMutexA(NULL, FALSE, NULL);
        /* FLAW: check for an error, but do nothing if one occurred */
        if (hMutex == NULL)
        {
            /* do nothing */
        }
        if (GetLastError() == ERROR_ALREADY_EXISTS)
        {
            /* do nothing */
        }
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hMutex);
    }
    }
}

void CWE390_Error_Without_Action__char_w32CreateMutex_15_good()
{
    good1();
    good2();
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
    CWE390_Error_Without_Action__char_w32CreateMutex_15_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE390_Error_Without_Action__char_w32CreateMutex_15_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
