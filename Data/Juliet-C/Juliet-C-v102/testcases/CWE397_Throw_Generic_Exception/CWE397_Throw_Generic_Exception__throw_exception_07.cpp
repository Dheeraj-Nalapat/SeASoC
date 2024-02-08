/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE397_Throw_Generic_Exception__throw_exception_07.cpp
Label Definition File: CWE397_Throw_Generic_Exception__throw_exception.label.xml
Template File: point-flaw-07.tmpl.cpp
*/
/*
 * @description
 * CWE: 397 Declaration of Throws for Generic Exception.  In this functional variant, we actually throw a generic exception.
 * Sinks: throw_exception
 *    GoodSink: Throw a specific exception (range_error)
 *    BadSink : Throw std::exception class, which is very generic
 * Flow Variant: 07 Control flow: if(static_five==5) and if(static_five!=5)
 *
 * */

#include "std_testcase.h"

#include <string>
#include <iostream>
#include <stdexcept>

using namespace std; // To use exception and range_error easily

/* The variable below is not declared "const", but is never assigned
   any other value so a tool should be able to identify that reads of
   this will always give its initialized value. */
static int static_five = 5;

namespace CWE397_Throw_Generic_Exception__throw_exception_07
{

#ifndef OMITBAD

void bad()
{
    if(static_five==5)
    {
        throw exception(); /* FLAW: throw std::exception class, which is very generic */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        throw range_error("Test"); /* FIX: Throw a specific exception */
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(static_five!=5) instead of if(static_five==5) */
static void good1()
{
    if(static_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        throw exception(); /* FLAW: throw std::exception class, which is very generic */
    }
    else
    {
        throw range_error("Test"); /* FIX: Throw a specific exception */
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_five==5)
    {
        throw range_error("Test"); /* FIX: Throw a specific exception */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        throw exception(); /* FLAW: throw std::exception class, which is very generic */
    }
}

void good()
{
    good1();
    good2();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE397_Throw_Generic_Exception__throw_exception_07; // so that we can use good and bad easily

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
