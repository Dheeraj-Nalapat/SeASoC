/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_12.c
Label Definition File: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free.label.xml
Template File: sources-sink-12.tmpl.c
*/
/*
 * @description
 * CWE: 590 Free of Invalid Pointer Not on the Heap
 * BadSource: alloca Data buffer is allocated on the stack with alloca()
 * GoodSource: Allocate memory on the heap
 * Sink:
 *    BadSink : Print then free data
 * Flow Variant: 12 Control flow: if(global_returns_t_or_f())
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_12_bad()
{
    int * data;
    data = NULL; /* Initialize data */
    if(global_returns_t_or_f())
    {
        {
            /* FLAW: data is allocated on the stack and deallocated in the BadSink */
            int * data_buf = (int *)ALLOCA(100*sizeof(int));
            {
                size_t i;
                for (i = 0; i < 100; i++)
                {
                    data_buf[i] = 5;
                }
            }
            data = data_buf;
        }
    }
    else
    {
        {
            /* FIX: data is allocated on the heap and deallocated in the BadSink */
            int * data_buf = (int *)malloc(100*sizeof(int));
            {
                size_t i;
                for (i = 0; i < 100; i++)
                {
                    data_buf[i] = 5;
                }
            }
            data = data_buf;
        }
    }
    printIntLine(data[0]);
    /* POTENTIAL FLAW: Possibly deallocating memory allocated on the stack */
    free(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() - use goodsource and badsink by changing the "if" so that
   both branches use the GoodSource */
static void goodG2B()
{
    int * data;
    data = NULL; /* Initialize data */
    if(global_returns_t_or_f())
    {
        {
            /* FIX: data is allocated on the heap and deallocated in the BadSink */
            int * data_buf = (int *)malloc(100*sizeof(int));
            {
                size_t i;
                for (i = 0; i < 100; i++)
                {
                    data_buf[i] = 5;
                }
            }
            data = data_buf;
        }
    }
    else
    {
        {
            /* FIX: data is allocated on the heap and deallocated in the BadSink */
            int * data_buf = (int *)malloc(100*sizeof(int));
            {
                size_t i;
                for (i = 0; i < 100; i++)
                {
                    data_buf[i] = 5;
                }
            }
            data = data_buf;
        }
    }
    printIntLine(data[0]);
    /* POTENTIAL FLAW: Possibly deallocating memory allocated on the stack */
    free(data);
}

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_12_good()
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
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_12_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_12_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
