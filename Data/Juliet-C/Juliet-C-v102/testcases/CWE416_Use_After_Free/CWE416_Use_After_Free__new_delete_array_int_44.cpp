/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE416_Use_After_Free__new_delete_array_int_44.cpp
Label Definition File: CWE416_Use_After_Free__new_delete_array.label.xml
Template File: sources-sinks-44.tmpl.cpp
*/
/*
 * @description
 * CWE: 416 Use After Free
 * BadSource:  Allocate data using new, initialize memory block, and Deallocate data using delete
 * GoodSource: Allocate data using new and initialize memory block
 * Sinks:
 *    GoodSink: Do nothing
 *    BadSink : Use data after free()
 * Flow Variant: 44 Data/control flow: data passed as an argument from one function to a function in the same source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE416_Use_After_Free__new_delete_array_int_44
{

#ifndef OMITBAD

static void bad_sink(int * data)
{
    /* POTENTIAL FLAW - Use of data that may have been deleted */
    printIntLine(data[0]);
    /* POTENTIAL INCIDENTAL - Possible memory leak here if data was not deleted */
}

void bad()
{
    int * data;
    /* define a function pointer */
    void (*func_ptr) (int *) = bad_sink;
    /* Initialize data */
    data = NULL;
    data = new int[100];
    {
        size_t i;
        for(i = 0; i < 100; i++)
        {
            data[i] = 5;
        }
    }
    /* POTENTIAL FLAW: Delete data in the source - the bad sink attempts to use data */
    delete [] data;
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_sink(int * data)
{
    /* POTENTIAL FLAW - Use of data that may have been deleted */
    printIntLine(data[0]);
    /* POTENTIAL INCIDENTAL - Possible memory leak here if data was not deleted */
}

static void goodG2B()
{
    int * data;
    void (*func_ptr) (int *) = goodG2B_sink;
    /* Initialize data */
    data = NULL;
    data = new int[100];
    {
        size_t i;
        for(i = 0; i < 100; i++)
        {
            data[i] = 5;
        }
    }
    /* FIX: Do not delete data in the source */
    func_ptr(data);
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G_sink(int * data)
{
    /* FIX - Don't use data that may have been deleted already */
    /* POTENTIAL INCIDENTAL - Possible memory leak here if data was not deleted */
    /* do nothing */
    ; /* empty statement needed for some flow variants */
}

static void goodB2G()
{
    int * data;
    void (*func_ptr) (int *) = goodB2G_sink;
    /* Initialize data */
    data = NULL;
    data = new int[100];
    {
        size_t i;
        for(i = 0; i < 100; i++)
        {
            data[i] = 5;
        }
    }
    /* POTENTIAL FLAW: Delete data in the source - the bad sink attempts to use data */
    delete [] data;
    func_ptr(data);
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

using namespace CWE416_Use_After_Free__new_delete_array_int_44; // so that we can use good and bad easily

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
