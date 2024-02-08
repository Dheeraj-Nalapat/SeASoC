/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__new_array_int_42.cpp
Label Definition File: CWE401_Memory_Leak__new_array.label.xml
Template File: sources-sinks-42.tmpl.cpp
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * BadSource:  Allocate data using new[]
 * GoodSource: Point data to a stack buffer
 * Sinks:
 *    GoodSink: call delete[] on data
 *    BadSink : no deallocation of data
 * Flow Variant: 42 Data flow: data returned from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

namespace CWE401_Memory_Leak__new_array_int_42
{

#ifndef OMITBAD

static int * bad_source(int * data)
{
    /* POTENTIAL FLAW: Allocate memory on the heap */
    data = new int[100];
    /* Initialize and make use of data */
    data[0] = 5;
    printIntLine(data[0]);
    return data;
}

void bad()
{
    int * data;
    data = NULL;
    data = bad_source(data);
    /* POTENTIAL FLAW: No deallocation */
    ; /* empty statement needed for some flow variants */
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static int * goodG2B_source(int * data)
{
    {
        /* FIX: Use memory allocated on the stack */
        int data_goodbuf[100];
        data = data_goodbuf;
        /* Initialize and make use of data */
        data[0] = 5;
        printIntLine(data[0]);
    }
    return data;
}

static void goodG2B()
{
    int * data;
    data = NULL;
    data = goodG2B_source(data);
    /* POTENTIAL FLAW: No deallocation */
    ; /* empty statement needed for some flow variants */
}

/* goodB2G() uses the BadSource with the GoodSink */
static int * goodB2G_source(int * data)
{
    /* POTENTIAL FLAW: Allocate memory on the heap */
    data = new int[100];
    /* Initialize and make use of data */
    data[0] = 5;
    printIntLine(data[0]);
    return data;
}

static void goodB2G()
{
    int * data;
    data = NULL;
    data = goodB2G_source(data);
    /* FIX: Deallocate memory */
    delete[] data;
}

void good()
{
    goodG2B();
    goodB2G();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE401_Memory_Leak__new_array_int_42; // so that we can use good and bad easily

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
