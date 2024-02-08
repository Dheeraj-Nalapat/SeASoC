/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE252_Unchecked_Return_Value__char_w32CreateNamedPipe_15.c
Label Definition File: CWE252_Unchecked_Return_Value.string.label.xml
Template File: point-flaw-15.tmpl.c
*/
/*
 * @description
 * CWE: 252 Unchecked Return Value
 * Sinks: w32CreateNamedPipe
 *    GoodSink: Check the return value of CreateNamedPipeA() and handle it properly
 *    BadSink : Do not check if CreateNamedPipeA() fails
 * Flow Variant: 15 Control flow: switch(6)
 *
 * */

#include "std_testcase.h"

#include <windows.h>
#define BUFSIZE 1024

#ifndef OMITBAD

void CWE252_Unchecked_Return_Value__char_w32CreateNamedPipe_15_bad()
{
    switch(6)
    {
    case 6:
    {
        char * pipeName = "\\\\.\\pipe\\mypipe";
        HANDLE hPipe = INVALID_HANDLE_VALUE;
        BOOL fConnected = FALSE;
        hPipe = CreateNamedPipeA(
                    pipeName,
                    FILE_FLAG_FIRST_PIPE_INSTANCE, /* FILE_FLAG_FIRST_PIPE_INSTANCE - this flag must be set */
                    PIPE_TYPE_MESSAGE |
                    PIPE_READMODE_MESSAGE |
                    PIPE_WAIT,
                    PIPE_UNLIMITED_INSTANCES,
                    BUFSIZE,
                    BUFSIZE,
                    NMPWAIT_USE_DEFAULT_WAIT,
                    NULL);
        /* FLAW: Do not check the return value */
        fConnected = ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hPipe);
    }
    break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        char * pipeName = "\\\\.\\pipe\\mypipe";
        HANDLE hPipe = INVALID_HANDLE_VALUE;
        BOOL fConnected = FALSE;
        hPipe = CreateNamedPipeA(
                    pipeName,
                    FILE_FLAG_FIRST_PIPE_INSTANCE, /* FILE_FLAG_FIRST_PIPE_INSTANCE - this flag must be set */
                    PIPE_TYPE_MESSAGE |
                    PIPE_READMODE_MESSAGE |
                    PIPE_WAIT,
                    PIPE_UNLIMITED_INSTANCES,
                    BUFSIZE,
                    BUFSIZE,
                    NMPWAIT_USE_DEFAULT_WAIT,
                    NULL);
        /* FIX: Check the return value of CreateNamedPipe() for an invalid handle AND
         * Check the return value of GetLastError() for ERROR_ACCESS_DENIED */
        if (hPipe == INVALID_HANDLE_VALUE)
        {
            exit(1);
        }
        if (GetLastError() == ERROR_ACCESS_DENIED)
        {
            exit(1);
        }
        fConnected = ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hPipe);
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
        char * pipeName = "\\\\.\\pipe\\mypipe";
        HANDLE hPipe = INVALID_HANDLE_VALUE;
        BOOL fConnected = FALSE;
        hPipe = CreateNamedPipeA(
                    pipeName,
                    FILE_FLAG_FIRST_PIPE_INSTANCE, /* FILE_FLAG_FIRST_PIPE_INSTANCE - this flag must be set */
                    PIPE_TYPE_MESSAGE |
                    PIPE_READMODE_MESSAGE |
                    PIPE_WAIT,
                    PIPE_UNLIMITED_INSTANCES,
                    BUFSIZE,
                    BUFSIZE,
                    NMPWAIT_USE_DEFAULT_WAIT,
                    NULL);
        /* FLAW: Do not check the return value */
        fConnected = ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hPipe);
    }
    break;
    default:
    {
        char * pipeName = "\\\\.\\pipe\\mypipe";
        HANDLE hPipe = INVALID_HANDLE_VALUE;
        BOOL fConnected = FALSE;
        hPipe = CreateNamedPipeA(
                    pipeName,
                    FILE_FLAG_FIRST_PIPE_INSTANCE, /* FILE_FLAG_FIRST_PIPE_INSTANCE - this flag must be set */
                    PIPE_TYPE_MESSAGE |
                    PIPE_READMODE_MESSAGE |
                    PIPE_WAIT,
                    PIPE_UNLIMITED_INSTANCES,
                    BUFSIZE,
                    BUFSIZE,
                    NMPWAIT_USE_DEFAULT_WAIT,
                    NULL);
        /* FIX: Check the return value of CreateNamedPipe() for an invalid handle AND
         * Check the return value of GetLastError() for ERROR_ACCESS_DENIED */
        if (hPipe == INVALID_HANDLE_VALUE)
        {
            exit(1);
        }
        if (GetLastError() == ERROR_ACCESS_DENIED)
        {
            exit(1);
        }
        fConnected = ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hPipe);
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
        char * pipeName = "\\\\.\\pipe\\mypipe";
        HANDLE hPipe = INVALID_HANDLE_VALUE;
        BOOL fConnected = FALSE;
        hPipe = CreateNamedPipeA(
                    pipeName,
                    FILE_FLAG_FIRST_PIPE_INSTANCE, /* FILE_FLAG_FIRST_PIPE_INSTANCE - this flag must be set */
                    PIPE_TYPE_MESSAGE |
                    PIPE_READMODE_MESSAGE |
                    PIPE_WAIT,
                    PIPE_UNLIMITED_INSTANCES,
                    BUFSIZE,
                    BUFSIZE,
                    NMPWAIT_USE_DEFAULT_WAIT,
                    NULL);
        /* FIX: Check the return value of CreateNamedPipe() for an invalid handle AND
         * Check the return value of GetLastError() for ERROR_ACCESS_DENIED */
        if (hPipe == INVALID_HANDLE_VALUE)
        {
            exit(1);
        }
        if (GetLastError() == ERROR_ACCESS_DENIED)
        {
            exit(1);
        }
        fConnected = ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hPipe);
    }
    break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        char * pipeName = "\\\\.\\pipe\\mypipe";
        HANDLE hPipe = INVALID_HANDLE_VALUE;
        BOOL fConnected = FALSE;
        hPipe = CreateNamedPipeA(
                    pipeName,
                    FILE_FLAG_FIRST_PIPE_INSTANCE, /* FILE_FLAG_FIRST_PIPE_INSTANCE - this flag must be set */
                    PIPE_TYPE_MESSAGE |
                    PIPE_READMODE_MESSAGE |
                    PIPE_WAIT,
                    PIPE_UNLIMITED_INSTANCES,
                    BUFSIZE,
                    BUFSIZE,
                    NMPWAIT_USE_DEFAULT_WAIT,
                    NULL);
        /* FLAW: Do not check the return value */
        fConnected = ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
        /* We'll leave out most of the implementation since it has nothing to do with the CWE
         * and since the checkers are looking for certain function calls anyway */
        CloseHandle(hPipe);
    }
    }
}

void CWE252_Unchecked_Return_Value__char_w32CreateNamedPipe_15_good()
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
    CWE252_Unchecked_Return_Value__char_w32CreateNamedPipe_15_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE252_Unchecked_Return_Value__char_w32CreateNamedPipe_15_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
