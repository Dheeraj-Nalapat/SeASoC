/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE762_Mismatched_Memory_Management_Routines__delete_struct_malloc_43.cpp
Label Definition File: CWE762_Mismatched_Memory_Management_Routines__delete.label.xml
Template File: sources-sinks-43.tmpl.cpp
*/
/*
 * @description
 * CWE: 762 Mismatched Memory Management Routines
 * BadSource: malloc Allocate data using malloc()
 * GoodSource: Allocate data using new
 * Sinks:
 *    GoodSink: Deallocate data using free()
 *    BadSink : Deallocate data using delete
 * Flow Variant: 43 Data flow: data flows using a C++ reference from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

namespace CWE762_Mismatched_Memory_Management_Routines__delete_struct_malloc_43
{

#ifndef OMITBAD

void bad_source(twoints * &data)
{
    /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
    data = (twoints *)malloc(100*sizeof(twoints));
}

void bad()
{
    twoints * data;
    /* Initialize data*/
    data = NULL;
    bad_source(data);
    /* POTENTIAL FLAW: Deallocate memory using delete - the source memory allocation function may
     * require a call to free() to deallocate the memory */
    delete data;
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_source(twoints * &data)
{
    /* FIX: Allocate memory from the heap using new */
    data = new twoints;
}

static void goodG2B()
{
    twoints * data;
    /* Initialize data*/
    data = NULL;
    goodG2B_source(data);
    /* POTENTIAL FLAW: Deallocate memory using delete - the source memory allocation function may
     * require a call to free() to deallocate the memory */
    delete data;
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G_source(twoints * &data)
{
    /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
    data = (twoints *)malloc(100*sizeof(twoints));
}

static void goodB2G()
{
    twoints * data;
    /* Initialize data*/
    data = NULL;
    goodB2G_source(data);
    /* FIX: Deallocate the memory using free() */
    free(data);
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

using namespace CWE762_Mismatched_Memory_Management_Routines__delete_struct_malloc_43; // so that we can use good and bad easily

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
