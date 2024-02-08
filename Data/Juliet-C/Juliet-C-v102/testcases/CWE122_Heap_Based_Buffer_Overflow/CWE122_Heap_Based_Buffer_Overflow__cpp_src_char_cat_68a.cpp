/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE122_Heap_Based_Buffer_Overflow__cpp_src_char_cat_68a.cpp
Label Definition File: CWE122_Heap_Based_Buffer_Overflow__cpp_src.label.xml
Template File: sources-sink-68a.tmpl.cpp
*/
/*
 * @description
 * CWE: 122 Heap Based Buffer Overflow
 * BadSource:  Initialize data as a large string
 * GoodSource: Initialize data as a small string
 * Sink: cat
 *    BadSink : Copy data to string using strcat
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

char * CWE122_Heap_Based_Buffer_Overflow__cpp_src_char_cat_68_bad_data;
char * CWE122_Heap_Based_Buffer_Overflow__cpp_src_char_cat_68_goodG2B_data;

namespace CWE122_Heap_Based_Buffer_Overflow__cpp_src_char_cat_68
{

#ifndef OMITBAD

/* bad function declaration */
void bad_sink();

void bad()
{
    char * data;
    data = new char[100];
    /* FLAW: Initialize data as a large buffer that is larger than the small buffer used in the sink */
    memset(data, 'A', 100-1); /* fill with 'A's */
    data[100-1] = '\0'; /* null terminate */
    CWE122_Heap_Based_Buffer_Overflow__cpp_src_char_cat_68_bad_data = data;
    bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declarations */
void goodG2B_sink();

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    data = new char[100];
    /* FIX: Initialize data as a small buffer that as small or smaller than the small buffer used in the sink */
    memset(data, 'A', 50-1); /* fill with 'A's */
    data[50-1] = '\0'; /* null terminate */
    CWE122_Heap_Based_Buffer_Overflow__cpp_src_char_cat_68_goodG2B_data = data;
    goodG2B_sink();
}

void good()
{
    goodG2B();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE122_Heap_Based_Buffer_Overflow__cpp_src_char_cat_68; // so that we can use good and bad easily

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
