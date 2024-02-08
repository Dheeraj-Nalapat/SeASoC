/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE672_Operation_on_Resource_After_Expiration_or_Release__basic_06.cpp
Label Definition File: CWE672_Operation_on_Resource_After_Expiration_or_Release__basic.label.xml
Template File: sources-sinks-06.tmpl.cpp
*/
/*
 * @description
 * CWE: 672 Operation on Resource After Expiration or Release
 * BadSource:  Add values to the list, including the number zero
 * GoodSource: Add value to the list that are not zero
 * Sinks:
 *    GoodSink: Iterate through the list without attempting to clear its contents
 *    BadSink : Iterate through the list, but clear the list if it contains a zero
 * Flow Variant: 06 Control flow: if(static_const_five==5) and if(static_const_five!=5)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#include <list>
#include <iostream>
using namespace std;

/* The variable below is declared "const", so a tool should be able
   to identify that reads of this will always give its initialized
   value. */
static const int static_const_five = 5;

namespace CWE672_Operation_on_Resource_After_Expiration_or_Release__basic_06
{

#ifndef OMITBAD

void bad()
{
    list<int> data;
    if(static_const_five==5)
    {
        /* POTENTIAL FLAW: Insert a zero into the list */
        data.push_back(100);
        data.push_back(0);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Insert non-zero values into the list */
        data.push_back(100);
        data.push_back(200);
    }
    if(static_const_five==5)
    {
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                if (!*i)
                {
                    data.clear();
                }
                /* POTENTIAL FLAW: Dereference the iterator, which may be invalid if data is cleared */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                /* FIX: Do not make any attempt to clear the list */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second static_const_five==5 to static_const_five!=5 */
static void goodB2G1()
{
    list<int> data;
    if(static_const_five==5)
    {
        /* POTENTIAL FLAW: Insert a zero into the list */
        data.push_back(100);
        data.push_back(0);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Insert non-zero values into the list */
        data.push_back(100);
        data.push_back(200);
    }
    if(static_const_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                if (!*i)
                {
                    data.clear();
                }
                /* POTENTIAL FLAW: Dereference the iterator, which may be invalid if data is cleared */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
    else
    {
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                /* FIX: Do not make any attempt to clear the list */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    list<int> data;
    if(static_const_five==5)
    {
        /* POTENTIAL FLAW: Insert a zero into the list */
        data.push_back(100);
        data.push_back(0);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Insert non-zero values into the list */
        data.push_back(100);
        data.push_back(200);
    }
    if(static_const_five==5)
    {
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                /* FIX: Do not make any attempt to clear the list */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                if (!*i)
                {
                    data.clear();
                }
                /* POTENTIAL FLAW: Dereference the iterator, which may be invalid if data is cleared */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first static_const_five==5 to static_const_five!=5 */
static void goodG2B1()
{
    list<int> data;
    if(static_const_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Insert a zero into the list */
        data.push_back(100);
        data.push_back(0);
    }
    else
    {
        /* FIX: Insert non-zero values into the list */
        data.push_back(100);
        data.push_back(200);
    }
    if(static_const_five==5)
    {
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                if (!*i)
                {
                    data.clear();
                }
                /* POTENTIAL FLAW: Dereference the iterator, which may be invalid if data is cleared */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                /* FIX: Do not make any attempt to clear the list */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    list<int> data;
    if(static_const_five==5)
    {
        /* FIX: Insert non-zero values into the list */
        data.push_back(100);
        data.push_back(200);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Insert a zero into the list */
        data.push_back(100);
        data.push_back(0);
    }
    if(static_const_five==5)
    {
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                if (!*i)
                {
                    data.clear();
                }
                /* POTENTIAL FLAW: Dereference the iterator, which may be invalid if data is cleared */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            list<int>::iterator i;
            cout << "The list contains: ";
            for( i = data.begin(); i != data.end(); i++)
            {
                /* FIX: Do not make any attempt to clear the list */
                cout << " " << *i;
            }
            cout << endl;
        }
    }
}

void good()
{
    goodB2G1();
    goodB2G2();
    goodG2B1();
    goodG2B2();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE672_Operation_on_Resource_After_Expiration_or_Release__basic_06; // so that we can use good and bad easily

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
