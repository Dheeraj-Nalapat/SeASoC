/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_alloca_65a.c
Label Definition File: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free.label.xml
Template File: sources-sink-65a.tmpl.c
*/
/*
 * @description
 * CWE: 590 Free of Invalid Pointer Not on the Heap
 * BadSource: alloca Data buffer is allocated on the stack with alloca()
 * GoodSource: Allocate memory on the heap
 * Sinks:
 *    BadSink : Print then free data
 * Flow Variant: 65 Data/control flow: data passed as an argument from one function to a function in a different source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_alloca_65b_bad_sink(wchar_t * data);

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_alloca_65_bad()
{
    wchar_t * data;
    /* define a function pointer */
    void (*func_ptr) (wchar_t *) = CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_alloca_65b_bad_sink;
    data = NULL; /* Initialize data */
    {
        /* FLAW: data is allocated on the stack and deallocated in the BadSink */
        wchar_t * data_buf = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
        wmemset(data_buf, L'A', 100-1); /* fill with 'A's */
        data_buf[100-1] = L'\0'; /* null terminate */
        data = data_buf;
    }
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_alloca_65b_goodG2B_sink(wchar_t * data);

static void goodG2B()
{
    wchar_t * data;
    void (*func_ptr) (wchar_t *) = CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_alloca_65b_goodG2B_sink;
    data = NULL; /* Initialize data */
    {
        /* FIX: data is allocated on the heap and deallocated in the BadSink */
        wchar_t * data_buf = (wchar_t *)malloc(100*sizeof(wchar_t));
        wmemset(data_buf, L'A', 100-1); /* fill with 'A's */
        data_buf[100-1] = L'\0'; /* null terminate */
        data = data_buf;
    }
    func_ptr(data);
}

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_alloca_65_good()
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
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_alloca_65_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_wchar_t_alloca_65_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
