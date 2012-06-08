def main():
    in_file = 'predictions.txt'
    out_file = 'to_submit.csv'

    with open(in_file, 'r') as f_in:
        lines = list(f_in)

    with open(out_file, 'w') as f_out:
        writeln = lambda s : f_out.write(s + '\n')
        writeln('source_node,destination_nodes')
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('P'):
                continue
            line = line[1:] # drop leading 'P'
            head, tail = line.split(':')
            head = head.strip()
            tail = tail.split()
            # return to 1-based indexing scheme
            head = str(int(head) + 1)
            tail = [str(int(x) + 1) for x in tail]
            writeln('%s,%s' % (head, ' '.join(tail)))

if __name__ == '__main__':
    main()
            
            
