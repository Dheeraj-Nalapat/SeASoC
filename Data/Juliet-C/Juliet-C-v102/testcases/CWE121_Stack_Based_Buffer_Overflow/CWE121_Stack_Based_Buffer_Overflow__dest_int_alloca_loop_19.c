/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_loop_19.c
Label Definition File: CWE121_Stack_Based_Buffer_Overflow__dest.label.xml
Template File: sources-sink-19.tmpl.c
*/
/*
 * @description
 * CWE: 121 Stack Based Buffer Overflow
 * BadSource:  Set data pointer to the bad buffer
 * GoodSource: Set data pointer to the good buffer
 * Sink: loop
 *    BadSink : Copy int array to data using a loop
 * Flow Variant: 19 Control flow: Dead code after a return
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_loop_19_bad()
{
    int * data;
    int * data_badbuf = (int *)ALLOCA(50*sizeof(int));
    int * data_goodbuf = (int *)ALLOCA(100*sizeof(int));
    /* FLAW: Set a pointer to a "small" buffer. This buffer will be used in the sinks as a destination
     * buffer in various memory copying functions using a "large" source buffer. */
    data = data_badbuf;
    {
        int src[100] = {0}; /* fill with 0's */
        {
            size_t i;
            /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
            for (i = 0; i < 100; i++)
            {
                data[i] = src[i];
            }
            printIntLine(data[0]);
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    /* FIX: Set a pointer to a "large" buffer, thus avoiding buffer overflows in the sinks. */
    data = data_goodbuf;
    {
        int src[100] = {0}; /* fill with 0's */
        {
            size_t i;
            /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
            for (i = 0; i < 100; i++)
            {
                data[i] = src[i];
            }
            printIntLine(data[0]);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() - use goodsource and badsink by swapping the GoodSource and BadSource around the return */
static void goodG2B()
{
    int * data;
    int * data_badbuf = (int *)ALLOCA(50*sizeof(int));
    int * data_goodbuf = (int *)ALLOCA(100*sizeof(int));
    /* FIX: Set a pointer to a "large" buffer, thus avoiding buffer overflows in the sinks. */
    data = data_goodbuf;
    {
        int src[100] = {0}; /* fill with 0's */
        {
            size_t i;
            /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
            for (i = 0; i < 100; i++)
            {
                data[i] = src[i];
            }
            printIntLine(data[0]);
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    /* FLAW: Set a pointer to a "small" buffer. This buffer will be used in the sinks as a destination
     * buffer in various memory copying functions using a "large" source buffer. */
    data = data_badbuf;
    {
        int src[100] = {0}; /* fill with 0's */
        {
            size_t i;
            /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
            for (i = 0; i < 100; i++)
            {
                data[i] = src[i];
            }
            printIntLine(data[0]);
        }
    }
}

void CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_loop_19_good()
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
    CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_loop_19_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE121_Stack_Based_Buffer_Overflow__dest_int_alloca_loop_19_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
