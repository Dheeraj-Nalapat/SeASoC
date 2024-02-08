/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68a.c
Label Definition File: CWE122_Heap_Based_Buffer_Overflow__c_dest.label.xml
Template File: sources-sink-68a.tmpl.c
*/
/*
 * @description
 * CWE: 122 Heap Based Buffer Overflow
 * BadSource:  Allocate using malloc() and set data pointer to a small buffer
 * GoodSource: Allocate using malloc() and set data pointer to a large buffer
 * Sink: memmove
 *    BadSink : Copy twoints array to data using memmove
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

twoints * CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68_bad_data;
twoints * CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68_goodG2B_data;

#ifndef OMITBAD

/* bad function declaration */
void CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68b_bad_sink();

void CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68_bad()
{
    twoints * data;
    data = NULL;
    /* FLAW: Allocate and point data to a small buffer that is smaller than the large buffer used in the sinks */
    data = (twoints *)malloc(50*sizeof(twoints));
    CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68_bad_data = data;
    CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68b_bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declarations */
void CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68b_goodG2B_sink();

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    twoints * data;
    data = NULL;
    /* FIX: Allocate and point data to a large buffer that is at least as large as the large buffer used in the sink */
    data = (twoints *)malloc(100*sizeof(twoints));
    CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68_goodG2B_data = data;
    CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68b_goodG2B_sink();
}

void CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68_good()
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
    CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE122_Heap_Based_Buffer_Overflow__c_dest_struct_memmove_68_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
