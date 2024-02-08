/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67a.c
Label Definition File: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free.label.xml
Template File: sources-sink-67a.tmpl.c
*/
/*
 * @description
 * CWE: 590 Free of Invalid Pointer Not on the Heap
 * BadSource: static Data buffer is declared static on the stack
 * GoodSource: Allocate memory on the heap
 * Sinks:
 *    BadSink : Print then free data
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

typedef struct _CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67_struct_type
{
    int * a;
} CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67_struct_type;

#ifndef OMITBAD

/* bad function declaration */
void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67b_bad_sink(CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67_struct_type my_struct);

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67_bad()
{
    int * data;
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67_struct_type my_struct;
    data = NULL; /* Initialize data */
    {
        /* FLAW: data is allocated on the stack and deallocated in the BadSink */
        static int data_buf[100];
        {
            size_t i;
            for (i = 0; i < 100; i++)
            {
                data_buf[i] = 5;
            }
        }
        data = data_buf;
    }
    my_struct.a = data;
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67b_bad_sink(my_struct);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67b_goodG2B_sink(CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67_struct_type my_struct);

static void goodG2B()
{
    int * data;
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67_struct_type my_struct;
    data = NULL; /* Initialize data */
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
    my_struct.a = data;
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67b_goodG2B_sink(my_struct);
}

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67_good()
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
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_static_67_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
