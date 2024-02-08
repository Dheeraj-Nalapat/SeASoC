/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memmove_61a.c
Label Definition File: CWE122_Heap_Based_Buffer_Overflow__c_dest.label.xml
Template File: sources-sink-61a.tmpl.c
*/
/*
 * @description
 * CWE: 122 Heap Based Buffer Overflow
 * BadSource:  Allocate using malloc() and set data pointer to a small buffer
 * GoodSource: Allocate using malloc() and set data pointer to a large buffer
 * Sinks: memmove
 *    BadSink : Copy long long array to data using memmove
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

/* bad function declaration */
long long * CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memmove_61b_bad_source(long long * data);

void CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memmove_61_bad()
{
    long long * data;
    data = NULL;
    data = CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memmove_61b_bad_source(data);
    {
        long long src[100] = {0}; /* fill with 0's */
        /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
        memmove(data, src, 100*sizeof(long long));
        printLongLongLine(data[0]);
        free(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
long long * CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memmove_61b_goodG2B_source(long long * data);

static void goodG2B()
{
    long long * data;
    data = NULL;
    data = CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memmove_61b_goodG2B_source(data);
    {
        long long src[100] = {0}; /* fill with 0's */
        /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
        memmove(data, src, 100*sizeof(long long));
        printLongLongLine(data[0]);
        free(data);
    }
}

void CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memmove_61_good()
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
    CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memmove_61_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE122_Heap_Based_Buffer_Overflow__c_dest_long_long_memmove_61_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
