/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE404_Improper_Resource_Shutdown__open_w32CloseHandle_15.c
Label Definition File: CWE404_Improper_Resource_Shutdown__open.label.xml
Template File: source-sinks-15.tmpl.c
*/
/*
 * @description
 * CWE: 404 Improper Resource Shutdown or Release
 * BadSource:  Open a file using open()
 * Sinks: w32CloseHandle
 *    GoodSink: Close the file using close()
 *    BadSink : Close the file using CloseHandle
 * Flow Variant: 15 Control flow: switch(6)
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define OPEN _open
# define CLOSE _close
#else
# define OPEN open
# define CLOSE close
#endif

#include <windows.h>

#ifndef OMITBAD

void CWE404_Improper_Resource_Shutdown__open_w32CloseHandle_15_bad()
{
    int data;
    /* Initialize data */
    data = -1;
    data = OPEN("BadSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
    switch(6)
    {
    case 6:
        if (data != -1)
        {
            /* FLAW: Attempt to close the file using CloseHandle() instead of close() */
            CloseHandle((HANDLE)data);
        }
        break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        if (data != -1)
        {
            /* FIX: Close the file using close() */
            CLOSE(data);
        }
        break;
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the switch to switch(5) */
static void goodB2G1()
{
    int data;
    /* Initialize data */
    data = -1;
    data = OPEN("BadSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
    switch(5)
    {
    case 6:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        if (data != -1)
        {
            /* FLAW: Attempt to close the file using CloseHandle() instead of close() */
            CloseHandle((HANDLE)data);
        }
        break;
    default:
        if (data != -1)
        {
            /* FIX: Close the file using close() */
            CLOSE(data);
        }
        break;
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the switch */
static void goodB2G2()
{
    int data;
    /* Initialize data */
    data = -1;
    data = OPEN("BadSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
    switch(6)
    {
    case 6:
        if (data != -1)
        {
            /* FIX: Close the file using close() */
            CLOSE(data);
        }
        break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        if (data != -1)
        {
            /* FLAW: Attempt to close the file using CloseHandle() instead of close() */
            CloseHandle((HANDLE)data);
        }
        break;
    }
}

void CWE404_Improper_Resource_Shutdown__open_w32CloseHandle_15_good()
{
    goodB2G1();
    goodB2G2();
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
    CWE404_Improper_Resource_Shutdown__open_w32CloseHandle_15_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE404_Improper_Resource_Shutdown__open_w32CloseHandle_15_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
