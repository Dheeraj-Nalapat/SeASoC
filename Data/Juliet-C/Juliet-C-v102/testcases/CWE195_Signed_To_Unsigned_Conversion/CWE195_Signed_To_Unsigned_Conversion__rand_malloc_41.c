/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE195_Signed_To_Unsigned_Conversion__rand_malloc_41.c
Label Definition File: CWE195_Signed_To_Unsigned_Conversion.label.xml
Template File: sources-sink-41.tmpl.c
*/
/*
 * @description
 * CWE: 195 Signed to Unsigned Conversion
 * BadSource: rand Set data to result of rand(), which may be zero
 * GoodSource: Positive integer
 * Sink: malloc
 *    BadSink : Allocate memory using malloc() with the size of data
 * Flow Variant: 41 Data flow: data passed as an argument from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE195_Signed_To_Unsigned_Conversion__rand_malloc_41_bad_sink(int data)
{
    {
        /* Assume we want to allocate a relatively small buffer */
        if (data < 100)
        {
            /* POTENTIAL FLAW: malloc() takes a size_t (unsigned int) as input and therefore if it is negative,
             * the coversion will cause malloc() to allocate a very large amount of data or fail */
            char * data_buf = (char *)malloc(data);
            /* Do something with data_buf */
            memset(data_buf, 'A', data-1);
            data_buf[data-1] = '\0';
            printLine(data_buf);
            free(data_buf);
        }
    }
}

void CWE195_Signed_To_Unsigned_Conversion__rand_malloc_41_bad()
{
    int data;
    /* Initialize data */
    data = -1;
    data = RAND32();
    CWE195_Signed_To_Unsigned_Conversion__rand_malloc_41_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

void CWE195_Signed_To_Unsigned_Conversion__rand_malloc_41_goodG2B_sink(int data)
{
    {
        /* Assume we want to allocate a relatively small buffer */
        if (data < 100)
        {
            /* POTENTIAL FLAW: malloc() takes a size_t (unsigned int) as input and therefore if it is negative,
             * the coversion will cause malloc() to allocate a very large amount of data or fail */
            char * data_buf = (char *)malloc(data);
            /* Do something with data_buf */
            memset(data_buf, 'A', data-1);
            data_buf[data-1] = '\0';
            printLine(data_buf);
            free(data_buf);
        }
    }
}

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    int data;
    /* Initialize data */
    data = -1;
    /* FIX: Use a positive integer less than &InitialDataSize&*/
    data = 100-1;
    CWE195_Signed_To_Unsigned_Conversion__rand_malloc_41_goodG2B_sink(data);
}

void CWE195_Signed_To_Unsigned_Conversion__rand_malloc_41_good()
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
    CWE195_Signed_To_Unsigned_Conversion__rand_malloc_41_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE195_Signed_To_Unsigned_Conversion__rand_malloc_41_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
