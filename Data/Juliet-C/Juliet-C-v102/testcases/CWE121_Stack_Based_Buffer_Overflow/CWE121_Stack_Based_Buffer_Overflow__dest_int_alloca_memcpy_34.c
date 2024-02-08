/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_memcpy_34.c
Label Definition File: CWE121_Stack_Based_Buffer_Overflow__dest.label.xml
Template File: sources-sink-34.tmpl.c
*/
/*
 * @description
 * CWE: 121 Stack Based Buffer Overflow
 * BadSource:  Set data pointer to the bad buffer
 * GoodSource: Set data pointer to the good buffer
 * Sinks: memcpy
 *    BadSink : Copy int array to data using memcpy
 * Flow Variant: 34 Data flow: use of a union containing two methods of accessing the same data (within the same function)
 *
 * */

#include "std_testcase.h"

typedef union
{
    int * a;
    int * b;
} CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_memcpy_34_union_type;

#ifndef OMITBAD

void CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_memcpy_34_bad()
{
    int * data;
    CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_memcpy_34_union_type my_union;
    int * data_badbuf = (int *)ALLOCA(50*sizeof(int));
    int * data_goodbuf = (int *)ALLOCA(100*sizeof(int));
    /* FLAW: Set a pointer to a "small" buffer. This buffer will be used in the sinks as a destination
     * buffer in various memory copying functions using a "large" source buffer. */
    data = data_badbuf;
    my_union.a = data;
    {
        int * data = my_union.b;
        {
            int src[100] = {0}; /* fill with 0's */
            /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
            memcpy(data, src, 100*sizeof(int));
            printIntLine(data[0]);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    int * data;
    CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_memcpy_34_union_type my_union;
    int * data_badbuf = (int *)ALLOCA(50*sizeof(int));
    int * data_goodbuf = (int *)ALLOCA(100*sizeof(int));
    /* FIX: Set a pointer to a "large" buffer, thus avoiding buffer overflows in the sinks. */
    data = data_goodbuf;
    my_union.a = data;
    {
        int * data = my_union.b;
        {
            int src[100] = {0}; /* fill with 0's */
            /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
            memcpy(data, src, 100*sizeof(int));
            printIntLine(data[0]);
        }
    }
}

void CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_memcpy_34_good()
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
    CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_memcpy_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_memcpy_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
