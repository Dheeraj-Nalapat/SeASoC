/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE195_Signed_To_Unsigned_Conversion__fscanf_memcpy_64a.c
Label Definition File: CWE195_Signed_To_Unsigned_Conversion.label.xml
Template File: sources-sink-64a.tmpl.c
*/
/*
 * @description
 * CWE: 195 Signed to Unsigned Conversion
 * BadSource: fscanf Read data from the console using fscanf()
 * GoodSource: Positive integer
 * Sinks: memcpy
 *    BadSink : Copy strings using memcpy() with the length of data
 * Flow Variant: 64 Data flow: void pointer to data passed from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

/* bad function declaration */
void CWE195_Signed_To_Unsigned_Conversion__fscanf_memcpy_64b_bad_sink(void * void_data_ptr);

void CWE195_Signed_To_Unsigned_Conversion__fscanf_memcpy_64_bad()
{
    int data;
    /* Initialize data */
    data = -1;
    fscanf (stdin, "%d", &data);
    CWE195_Signed_To_Unsigned_Conversion__fscanf_memcpy_64b_bad_sink(&data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE195_Signed_To_Unsigned_Conversion__fscanf_memcpy_64b_goodG2B_sink(void * void_data_ptr);

static void goodG2B()
{
    int data;
    /* Initialize data */
    data = -1;
    /* FIX: Use a positive integer less than &InitialDataSize&*/
    data = 100-1;
    CWE195_Signed_To_Unsigned_Conversion__fscanf_memcpy_64b_goodG2B_sink(&data);
}

void CWE195_Signed_To_Unsigned_Conversion__fscanf_memcpy_64_good()
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
    CWE195_Signed_To_Unsigned_Conversion__fscanf_memcpy_64_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE195_Signed_To_Unsigned_Conversion__fscanf_memcpy_64_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
