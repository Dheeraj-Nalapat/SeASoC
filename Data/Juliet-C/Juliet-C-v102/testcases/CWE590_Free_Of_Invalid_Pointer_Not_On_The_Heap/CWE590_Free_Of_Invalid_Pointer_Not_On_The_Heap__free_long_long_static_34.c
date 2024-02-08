/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_long_long_static_34.c
Label Definition File: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free.label.xml
Template File: sources-sink-34.tmpl.c
*/
/*
 * @description
 * CWE: 590 Free of Invalid Pointer Not on the Heap
 * BadSource: static Data buffer is declared static on the stack
 * GoodSource: Allocate memory on the heap
 * Sinks:
 *    BadSink : Print then free data
 * Flow Variant: 34 Data flow: use of a union containing two methods of accessing the same data (within the same function)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

typedef union
{
    long long * a;
    long long * b;
} CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_long_long_static_34_union_type;

#ifndef OMITBAD

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_long_long_static_34_bad()
{
    long long * data;
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_long_long_static_34_union_type my_union;
    data = NULL; /* Initialize data */
    {
        /* FLAW: data is allocated on the stack and deallocated in the BadSink */
        static long long data_buf[100];
        {
            size_t i;
            for (i = 0; i < 100; i++)
            {
                data_buf[i] = 5L;
            }
        }
        data = data_buf;
    }
    my_union.a = data;
    {
        long long * data = my_union.b;
        printLongLongLine(data[0]);
        /* POTENTIAL FLAW: Possibly deallocating memory allocated on the stack */
        free(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    long long * data;
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_long_long_static_34_union_type my_union;
    data = NULL; /* Initialize data */
    {
        /* FIX: data is allocated on the heap and deallocated in the BadSink */
        long long * data_buf = (long long *)malloc(100*sizeof(long long));
        {
            size_t i;
            for (i = 0; i < 100; i++)
            {
                data_buf[i] = 5L;
            }
        }
        data = data_buf;
    }
    my_union.a = data;
    {
        long long * data = my_union.b;
        printLongLongLine(data[0]);
        /* POTENTIAL FLAW: Possibly deallocating memory allocated on the stack */
        free(data);
    }
}

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_long_long_static_34_good()
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
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_long_long_static_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_long_long_static_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
