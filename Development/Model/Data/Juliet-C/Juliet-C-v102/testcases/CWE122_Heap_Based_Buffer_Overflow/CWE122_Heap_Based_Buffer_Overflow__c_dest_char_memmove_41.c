/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE122_Heap_Based_Buffer_Overflow__c_dest_char_memmove_41.c
Label Definition File: CWE122_Heap_Based_Buffer_Overflow__c_dest.string.label.xml
Template File: sources-sink-41.tmpl.c
*/
/*
 * @description
 * CWE: 122 Heap Based Buffer Overflow
 * BadSource:  Allocate using malloc() and set data pointer to a small buffer
 * GoodSource: Allocate using malloc() and set data pointer to a large buffer
 * Sink: memmove
 *    BadSink : Copy string to data using memmove
 * Flow Variant: 41 Data flow: data passed as an argument from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE122_Heap_Based_Buffer_Overflow__c_dest_char_memmove_41_bad_sink(char * data)
{
    {
        char src[100];
        memset(src, 'C', 100-1); /* fill with 'C's */
        src[100-1] = '\0'; /* null terminate */
        /* POTENTIAL FLAW: Possible buffer overflow if src is larger than data */
        memmove(data, src, 100*sizeof(char));
        data[100-1] = '\0'; /* Ensure the destination buffer is null terminated */
        printLine(data);
        free(data);
    }
}

void CWE122_Heap_Based_Buffer_Overflow__c_dest_char_memmove_41_bad()
{
    char * data;
    data = NULL;
    /* FLAW: Allocate and point data to a small buffer that is smaller than the large buffer used in the sinks */
    data = (char *)malloc(50*sizeof(char));
    data[0] = '\0'; /* null terminate */
    CWE122_Heap_Based_Buffer_Overflow__c_dest_char_memmove_41_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

void CWE122_Heap_Based_Buffer_Overflow__c_dest_char_memmove_41_goodG2B_sink(char * data)
{
    {
        char src[100];
        memset(src, 'C', 100-1); /* fill with 'C's */
        src[100-1] = '\0'; /* null terminate */
        /* POTENTIAL FLAW: Possible buffer overflow if src is larger than data */
        memmove(data, src, 100*sizeof(char));
        data[100-1] = '\0'; /* Ensure the destination buffer is null terminated */
        printLine(data);
        free(data);
    }
}

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    data = NULL;
    /* FIX: Allocate and point data to a large buffer that is at least as large as the large buffer used in the sink */
    data = (char *)malloc(100*sizeof(char));
    data[0] = '\0'; /* null terminate */
    CWE122_Heap_Based_Buffer_Overflow__c_dest_char_memmove_41_goodG2B_sink(data);
}

void CWE122_Heap_Based_Buffer_Overflow__c_dest_char_memmove_41_good()
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
    CWE122_Heap_Based_Buffer_Overflow__c_dest_char_memmove_41_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE122_Heap_Based_Buffer_Overflow__c_dest_char_memmove_41_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
