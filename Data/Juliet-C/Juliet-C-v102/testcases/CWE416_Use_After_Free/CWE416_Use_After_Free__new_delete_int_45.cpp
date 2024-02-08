/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE416_Use_After_Free__new_delete_int_45.cpp
Label Definition File: CWE416_Use_After_Free__new_delete.label.xml
Template File: sources-sinks-45.tmpl.cpp
*/
/*
 * @description
 * CWE: 416 Use After Free
 * BadSource:  Allocate data using new, initialize memory block, and Deallocate data using delete
 * GoodSource: Allocate data using new and initialize memory block
 * Sinks:
 *    GoodSink: Do nothing
 *    BadSink : Use data after free()
 * Flow Variant: 45 Data flow: data passed as a static global variable from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE416_Use_After_Free__new_delete_int_45
{

static int * bad_data;
static int * goodG2B_data;
static int * goodB2G_data;

#ifndef OMITBAD

static void bad_sink()
{
    int * data = bad_data;
    /* POTENTIAL FLAW - Use of data that may have been deleted */
    printIntLine(*data);
    /* POTENTIAL INCIDENTAL - Possible memory leak here if data was not deleted */
}

void bad()
{
    int * data;
    /* Initialize data */
    data = NULL;
    data = new int;
    *data = 5;
    /* POTENTIAL FLAW: Delete data in the source - the bad sink attempts to use data */
    delete data;
    bad_data = data;
    bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_sink()
{
    int * data = goodG2B_data;
    /* POTENTIAL FLAW - Use of data that may have been deleted */
    printIntLine(*data);
    /* POTENTIAL INCIDENTAL - Possible memory leak here if data was not deleted */
}

static void goodG2B()
{
    int * data;
    /* Initialize data */
    data = NULL;
    data = new int;
    *data = 5;
    /* FIX: Do not delete data in the source */
    goodG2B_data = data;
    goodG2B_sink();
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G_sink()
{
    int * data = goodB2G_data;
    /* FIX - Don't use data that may have been deleted already */
    /* POTENTIAL INCIDENTAL - Possible memory leak here if data was not deleted */
    /* do nothing */
    ; /* empty statement needed for some flow variants */
}

static void goodB2G()
{
    int * data;
    /* Initialize data */
    data = NULL;
    data = new int;
    *data = 5;
    /* POTENTIAL FLAW: Delete data in the source - the bad sink attempts to use data */
    delete data;
    goodB2G_data = data;
    goodB2G_sink();
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

using namespace CWE416_Use_After_Free__new_delete_int_45; // so that we can use good and bad easily

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
