/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memcpy_15.c
Label Definition File: CWE122_Heap_Based_Buffer_Overflow__c_dest.label.xml
Template File: sources-sink-15.tmpl.c
*/
/*
 * @description
 * CWE: 122 Heap Based Buffer Overflow
 * BadSource:  Allocate using malloc() and set data pointer to a small buffer
 * GoodSource: Allocate using malloc() and set data pointer to a large buffer
 * Sink: memcpy
 *    BadSink : Copy long long array to data using memcpy
 * Flow Variant: 15 Control flow: switch(6)
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memcpy_15_bad()
{
    long long * data;
    data = NULL;
    switch(6)
    {
    case 6:
        /* FLAW: Allocate and point data to a small buffer that is smaller than the large buffer used in the sinks */
        data = (long long *)malloc(50*sizeof(long long));
        break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Allocate and point data to a large buffer that is at least as large as the large buffer used in the sink */
        data = (long long *)malloc(100*sizeof(long long));
        break;
    }
    {
        long long src[100] = {0}; /* fill with 0's */
        /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
        memcpy(data, src, 100*sizeof(long long));
        printLongLongLine(data[0]);
        free(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B1() - use goodsource and badsink by changing the switch to switch(5) */
static void goodG2B1()
{
    long long * data;
    data = NULL;
    switch(5)
    {
    case 6:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Allocate and point data to a small buffer that is smaller than the large buffer used in the sinks */
        data = (long long *)malloc(50*sizeof(long long));
        break;
    default:
        /* FIX: Allocate and point data to a large buffer that is at least as large as the large buffer used in the sink */
        data = (long long *)malloc(100*sizeof(long long));
        break;
    }
    {
        long long src[100] = {0}; /* fill with 0's */
        /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
        memcpy(data, src, 100*sizeof(long long));
        printLongLongLine(data[0]);
        free(data);
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the switch */
static void goodG2B2()
{
    long long * data;
    data = NULL;
    switch(6)
    {
    case 6:
        /* FIX: Allocate and point data to a large buffer that is at least as large as the large buffer used in the sink */
        data = (long long *)malloc(100*sizeof(long long));
        break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Allocate and point data to a small buffer that is smaller than the large buffer used in the sinks */
        data = (long long *)malloc(50*sizeof(long long));
        break;
    }
    {
        long long src[100] = {0}; /* fill with 0's */
        /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
        memcpy(data, src, 100*sizeof(long long));
        printLongLongLine(data[0]);
        free(data);
    }
}

void CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memcpy_15_good()
{
    goodG2B1();
    goodG2B2();
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
    CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memcpy_15_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memcpy_15_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
