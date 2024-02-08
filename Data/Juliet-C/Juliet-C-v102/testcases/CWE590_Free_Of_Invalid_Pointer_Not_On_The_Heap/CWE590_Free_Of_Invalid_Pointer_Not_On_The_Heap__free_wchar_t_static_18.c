/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_static_18.c
Label Definition File: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free.label.xml
Template File: sources-sink-18.tmpl.c
*/
/*
 * @description
 * CWE: 590 Free of Invalid Pointer Not on the Heap
 * BadSource: static Data buffer is declared static on the stack
 * GoodSource: Allocate memory on the heap
 * Sink:
 *    BadSink : Print then free data
 * Flow Variant: 18 Control flow: goto statements
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_static_18_bad()
{
    wchar_t * data;
    data = NULL; /* Initialize data */
    goto source;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* FIX: data is allocated on the heap and deallocated in the BadSink */
        wchar_t * data_buf = (wchar_t *)malloc(100*sizeof(wchar_t));
        wmemset(data_buf, L'A', 100-1); /* fill with 'A's */
        data_buf[100-1] = L'\0'; /* null terminate */
        data = data_buf;
    }
source:
    {
        /* FLAW: data is allocated on the stack and deallocated in the BadSink */
        static wchar_t data_buf[100];
        wmemset(data_buf, L'A', 100-1); /* fill with 'A's */
        data_buf[100-1] = L'\0'; /* null terminate */
        data = data_buf;
    }
    printWLine(data);
    /* POTENTIAL FLAW: Possibly deallocating memory allocated on the stack */
    free(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() - use goodsource and badsink by reversing the blocks on the goto statement */
static void goodG2B()
{
    wchar_t * data;
    data = NULL; /* Initialize data */
    goto source;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* FLAW: data is allocated on the stack and deallocated in the BadSink */
        static wchar_t data_buf[100];
        wmemset(data_buf, L'A', 100-1); /* fill with 'A's */
        data_buf[100-1] = L'\0'; /* null terminate */
        data = data_buf;
    }
source:
    {
        /* FIX: data is allocated on the heap and deallocated in the BadSink */
        wchar_t * data_buf = (wchar_t *)malloc(100*sizeof(wchar_t));
        wmemset(data_buf, L'A', 100-1); /* fill with 'A's */
        data_buf[100-1] = L'\0'; /* null terminate */
        data = data_buf;
    }
    printWLine(data);
    /* POTENTIAL FLAW: Possibly deallocating memory allocated on the stack */
    free(data);
}

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_static_18_good()
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
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_static_18_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_static_18_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
