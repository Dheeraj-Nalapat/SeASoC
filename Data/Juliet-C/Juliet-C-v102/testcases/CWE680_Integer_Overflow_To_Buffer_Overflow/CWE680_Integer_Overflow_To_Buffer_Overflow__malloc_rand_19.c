/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE680_Integer_Overflow_To_Buffer_Overflow__malloc_rand_19.c
Label Definition File: CWE680_Integer_Overflow_To_Buffer_Overflow__malloc.label.xml
Template File: sources-sink-19.tmpl.c
*/
/*
 * @description
 * CWE: 680 Integer Overflow to Buffer Overflow
 * BadSource: rand Set data to result of rand(), which may be zero
 * GoodSource: Small number greater than zero that will not cause an integer overflow in the sink
 * Sink:
 *    BadSink : Attempt to allocate array using length value from source
 * Flow Variant: 19 Control flow: Dead code after a return
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE680_Integer_Overflow_To_Buffer_Overflow__malloc_rand_19_bad()
{
    int data;
    /* Initialize data */
    data = -1;
    data = RAND32();
    {
        size_t i;
        int *b;
        /* POTENTIAL FLAW: if data * sizeof(int) > SIZE_MAX, overflows to a small value
             so that the for loop doing the initialization causes a buffer overflow */
        b = (int*)malloc(data * sizeof(int));
        for (i = 0; i < (size_t)data; i++)
        {
            b[i] = 0; /* Potentially writes beyond the boundary of b */
        }
        printIntLine(b[0]);
        free(b);
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    /* FIX: Set data to a relatively small number greater than zero */
    data = 20;
    {
        size_t i;
        int *b;
        /* POTENTIAL FLAW: if data * sizeof(int) > SIZE_MAX, overflows to a small value
             so that the for loop doing the initialization causes a buffer overflow */
        b = (int*)malloc(data * sizeof(int));
        for (i = 0; i < (size_t)data; i++)
        {
            b[i] = 0; /* Potentially writes beyond the boundary of b */
        }
        printIntLine(b[0]);
        free(b);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() - use goodsource and badsink by swapping the GoodSource and BadSource around the return */
static void goodG2B()
{
    int data;
    /* Initialize data */
    data = -1;
    /* FIX: Set data to a relatively small number greater than zero */
    data = 20;
    {
        size_t i;
        int *b;
        /* POTENTIAL FLAW: if data * sizeof(int) > SIZE_MAX, overflows to a small value
             so that the for loop doing the initialization causes a buffer overflow */
        b = (int*)malloc(data * sizeof(int));
        for (i = 0; i < (size_t)data; i++)
        {
            b[i] = 0; /* Potentially writes beyond the boundary of b */
        }
        printIntLine(b[0]);
        free(b);
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    data = RAND32();
    {
        size_t i;
        int *b;
        /* POTENTIAL FLAW: if data * sizeof(int) > SIZE_MAX, overflows to a small value
             so that the for loop doing the initialization causes a buffer overflow */
        b = (int*)malloc(data * sizeof(int));
        for (i = 0; i < (size_t)data; i++)
        {
            b[i] = 0; /* Potentially writes beyond the boundary of b */
        }
        printIntLine(b[0]);
        free(b);
    }
}

void CWE680_Integer_Overflow_To_Buffer_Overflow__malloc_rand_19_good()
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
    CWE680_Integer_Overflow_To_Buffer_Overflow__malloc_rand_19_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE680_Integer_Overflow_To_Buffer_Overflow__malloc_rand_19_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
