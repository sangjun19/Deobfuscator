	.file	"kadwani_lab1b_3_flatten.c"
	.text
	.globl	_TIG_IZ_XXjf_envp
	.bss
	.align 8
	.type	_TIG_IZ_XXjf_envp, @object
	.size	_TIG_IZ_XXjf_envp, 8
_TIG_IZ_XXjf_envp:
	.zero	8
	.globl	isFileRead
	.align 4
	.type	isFileRead, @object
	.size	isFileRead, 4
isFileRead:
	.zero	4
	.globl	isExists
	.align 32
	.type	isExists, @object
	.size	isExists, 104
isExists:
	.zero	104
	.globl	_TIG_IZ_XXjf_argc
	.align 4
	.type	_TIG_IZ_XXjf_argc, @object
	.size	_TIG_IZ_XXjf_argc, 4
_TIG_IZ_XXjf_argc:
	.zero	4
	.globl	dist
	.align 32
	.type	dist, @object
	.size	dist, 104
dist:
	.zero	104
	.globl	adjacencyMatrix
	.align 32
	.type	adjacencyMatrix, @object
	.size	adjacencyMatrix, 2704
adjacencyMatrix:
	.zero	2704
	.globl	_TIG_IZ_XXjf_argv
	.align 8
	.type	_TIG_IZ_XXjf_argv, @object
	.size	_TIG_IZ_XXjf_argv, 8
_TIG_IZ_XXjf_argv:
	.zero	8
	.globl	parentArr
	.align 32
	.type	parentArr, @object
	.size	parentArr, 104
parentArr:
	.zero	104
	.section	.rodata
.LC0:
	.string	"   %c\t"
.LC1:
	.string	"Adjacency Matrix:\n\n\t"
.LC2:
	.string	"%c\t"
.LC3:
	.string	"   -\t"
.LC4:
	.string	"%4d\t"
.LC5:
	.string	"First read an input file!"
	.text
	.globl	showAdjacencyMatrix
	.type	showAdjacencyMatrix, @function
showAdjacencyMatrix:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$4, -8(%rbp)
.L43:
	cmpq	$28, -8(%rbp)
	ja	.L44
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L24-.L4
	.long	.L23-.L4
	.long	.L45-.L4
	.long	.L44-.L4
	.long	.L21-.L4
	.long	.L44-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L44-.L4
	.long	.L45-.L4
	.long	.L17-.L4
	.long	.L44-.L4
	.long	.L44-.L4
	.long	.L16-.L4
	.long	.L44-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L44-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L44-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L7:
	movl	-16(%rbp), %eax
	addl	$65, %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L25
.L21:
	movl	isFileRead(%rip), %eax
	testl	%eax, %eax
	jne	.L26
	movq	$17, -8(%rbp)
	jmp	.L25
.L26:
	movq	$15, -8(%rbp)
	jmp	.L25
.L15:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -16(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L25
.L23:
	movl	$10, %edi
	call	putchar@PLT
	movq	$20, -8(%rbp)
	jmp	.L25
.L14:
	movl	-16(%rbp), %eax
	addl	$65, %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -12(%rbp)
	movq	$28, -8(%rbp)
	jmp	.L25
.L8:
	addl	$1, -12(%rbp)
	movq	$28, -8(%rbp)
	jmp	.L25
.L10:
	movl	-12(%rbp), %eax
	movslq	%eax, %rcx
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rdx
	leaq	adjacencyMatrix(%rip), %rax
	movl	(%rdx,%rax), %eax
	cmpl	$-1, %eax
	jne	.L28
	movq	$26, -8(%rbp)
	jmp	.L25
.L28:
	movq	$19, -8(%rbp)
	jmp	.L25
.L6:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$24, -8(%rbp)
	jmp	.L25
.L16:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	isExists(%rip), %rax
	movl	(%rdx,%rax), %eax
	testl	%eax, %eax
	je	.L31
	movq	$16, -8(%rbp)
	jmp	.L25
.L31:
	movq	$20, -8(%rbp)
	jmp	.L25
.L12:
	movl	-12(%rbp), %eax
	movslq	%eax, %rcx
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rdx
	leaq	adjacencyMatrix(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$24, -8(%rbp)
	jmp	.L25
.L13:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L25
.L20:
	cmpl	$25, -16(%rbp)
	jg	.L33
	movq	$13, -8(%rbp)
	jmp	.L25
.L33:
	movq	$9, -8(%rbp)
	jmp	.L25
.L5:
	cmpl	$25, -16(%rbp)
	jg	.L35
	movq	$22, -8(%rbp)
	jmp	.L25
.L35:
	movq	$0, -8(%rbp)
	jmp	.L25
.L9:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	isExists(%rip), %rax
	movl	(%rdx,%rax), %eax
	testl	%eax, %eax
	je	.L37
	movq	$25, -8(%rbp)
	jmp	.L25
.L37:
	movq	$10, -8(%rbp)
	jmp	.L25
.L3:
	cmpl	$25, -12(%rbp)
	jg	.L39
	movq	$7, -8(%rbp)
	jmp	.L25
.L39:
	movq	$1, -8(%rbp)
	jmp	.L25
.L17:
	addl	$1, -16(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L25
.L24:
	movl	$10, %edi
	call	putchar@PLT
	movl	$0, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L25
.L19:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	isExists(%rip), %rax
	movl	(%rdx,%rax), %eax
	testl	%eax, %eax
	je	.L41
	movq	$21, -8(%rbp)
	jmp	.L25
.L41:
	movq	$24, -8(%rbp)
	jmp	.L25
.L11:
	addl	$1, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L25
.L44:
	nop
.L25:
	jmp	.L43
.L45:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	showAdjacencyMatrix, .-showAdjacencyMatrix
	.section	.rodata
.LC6:
	.string	" %c%c%c%c%d"
.LC7:
	.string	"%s successfully read!\n"
.LC8:
	.string	"Enter the file name: "
.LC9:
	.string	"%s"
.LC10:
	.string	"r"
	.text
	.globl	readInputFile
	.type	readInputFile, @function
readInputFile:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$336, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -288(%rbp)
.L74:
	cmpq	$24, -288(%rbp)
	ja	.L77
	movq	-288(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L49(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L49(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L49:
	.long	.L77-.L49
	.long	.L63-.L49
	.long	.L62-.L49
	.long	.L61-.L49
	.long	.L60-.L49
	.long	.L59-.L49
	.long	.L77-.L49
	.long	.L77-.L49
	.long	.L58-.L49
	.long	.L77-.L49
	.long	.L57-.L49
	.long	.L56-.L49
	.long	.L55-.L49
	.long	.L77-.L49
	.long	.L54-.L49
	.long	.L53-.L49
	.long	.L52-.L49
	.long	.L51-.L49
	.long	.L77-.L49
	.long	.L77-.L49
	.long	.L77-.L49
	.long	.L77-.L49
	.long	.L77-.L49
	.long	.L78-.L49
	.long	.L48-.L49
	.text
.L60:
	movl	-320(%rbp), %eax
	movslq	%eax, %rcx
	movl	-324(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rdx
	leaq	adjacencyMatrix(%rip), %rax
	movl	$-1, (%rdx,%rax)
	addl	$1, -320(%rbp)
	movq	$24, -288(%rbp)
	jmp	.L64
.L54:
	movzbl	-331(%rbp), %eax
	movsbl	%al, %eax
	subl	$65, %eax
	movl	%eax, -312(%rbp)
	movzbl	-330(%rbp), %eax
	movsbl	%al, %eax
	subl	$65, %eax
	movl	%eax, -308(%rbp)
	movl	$1, -304(%rbp)
	movl	-308(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	isExists(%rip), %rdx
	movl	-304(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movl	-312(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	isExists(%rip), %rdx
	movl	-304(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movl	-328(%rbp), %eax
	movl	%eax, -300(%rbp)
	movl	-312(%rbp), %eax
	movslq	%eax, %rcx
	movl	-308(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rcx
	leaq	adjacencyMatrix(%rip), %rdx
	movl	-300(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movl	-308(%rbp), %eax
	movslq	%eax, %rcx
	movl	-312(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rcx
	leaq	adjacencyMatrix(%rip), %rdx
	movl	-300(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$8, -288(%rbp)
	jmp	.L64
.L53:
	addl	$1, -324(%rbp)
	movq	$1, -288(%rbp)
	jmp	.L64
.L55:
	movl	$0, -320(%rbp)
	movq	$24, -288(%rbp)
	jmp	.L64
.L58:
	leaq	-329(%rbp), %r8
	leaq	-330(%rbp), %rdi
	leaq	-329(%rbp), %rcx
	leaq	-331(%rbp), %rdx
	movq	-296(%rbp), %rax
	subq	$8, %rsp
	leaq	-328(%rbp), %rsi
	pushq	%rsi
	movq	%r8, %r9
	movq	%rdi, %r8
	leaq	.LC6(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	addq	$16, %rsp
	movl	%eax, -316(%rbp)
	movq	$16, -288(%rbp)
	jmp	.L64
.L63:
	cmpl	$25, -324(%rbp)
	jg	.L65
	movq	$12, -288(%rbp)
	jmp	.L64
.L65:
	movq	$5, -288(%rbp)
	jmp	.L64
.L61:
	movl	$0, -324(%rbp)
	movq	$2, -288(%rbp)
	jmp	.L64
.L52:
	cmpl	$5, -316(%rbp)
	jne	.L68
	movq	$14, -288(%rbp)
	jmp	.L64
.L68:
	movq	$11, -288(%rbp)
	jmp	.L64
.L48:
	cmpl	$25, -320(%rbp)
	jg	.L70
	movq	$4, -288(%rbp)
	jmp	.L64
.L70:
	movq	$15, -288(%rbp)
	jmp	.L64
.L56:
	movq	-296(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	$1, isFileRead(%rip)
	leaq	-272(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$23, -288(%rbp)
	jmp	.L64
.L51:
	movl	-324(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	isExists(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -324(%rbp)
	movq	$2, -288(%rbp)
	jmp	.L64
.L59:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-272(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-272(%rbp), %rax
	leaq	.LC10(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -280(%rbp)
	movq	-280(%rbp), %rax
	movq	%rax, -296(%rbp)
	movq	$8, -288(%rbp)
	jmp	.L64
.L57:
	movl	$0, -324(%rbp)
	movq	$1, -288(%rbp)
	jmp	.L64
.L62:
	cmpl	$25, -324(%rbp)
	jg	.L72
	movq	$17, -288(%rbp)
	jmp	.L64
.L72:
	movq	$10, -288(%rbp)
	jmp	.L64
.L77:
	nop
.L64:
	jmp	.L74
.L78:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L76
	call	__stack_chk_fail@PLT
.L76:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	readInputFile, .-readInputFile
	.section	.rodata
	.align 8
.LC11:
	.string	"\nThere is no path from %c to %c\n"
.LC12:
	.string	"Enter the source vertex: "
.LC13:
	.string	" %c"
	.align 8
.LC14:
	.string	"Enter the destination vertex: "
	.align 8
.LC15:
	.string	"\nThe shortest path from %c to %c: "
.LC16:
	.string	"%c "
.LC17:
	.string	"\nThe length of this path: %d\n"
	.text
	.globl	shortestPath
	.type	shortestPath, @function
shortestPath:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$27, -48(%rbp)
.L121:
	cmpq	$33, -48(%rbp)
	ja	.L124
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L82(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L82(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L82:
	.long	.L102-.L82
	.long	.L101-.L82
	.long	.L100-.L82
	.long	.L124-.L82
	.long	.L99-.L82
	.long	.L124-.L82
	.long	.L124-.L82
	.long	.L124-.L82
	.long	.L124-.L82
	.long	.L124-.L82
	.long	.L98-.L82
	.long	.L97-.L82
	.long	.L96-.L82
	.long	.L125-.L82
	.long	.L125-.L82
	.long	.L124-.L82
	.long	.L93-.L82
	.long	.L92-.L82
	.long	.L91-.L82
	.long	.L90-.L82
	.long	.L124-.L82
	.long	.L89-.L82
	.long	.L88-.L82
	.long	.L124-.L82
	.long	.L124-.L82
	.long	.L87-.L82
	.long	.L124-.L82
	.long	.L86-.L82
	.long	.L124-.L82
	.long	.L85-.L82
	.long	.L84-.L82
	.long	.L83-.L82
	.long	.L124-.L82
	.long	.L81-.L82
	.text
.L91:
	movzbl	-77(%rbp), %eax
	movsbl	%al, %edx
	movzbl	-78(%rbp), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -48(%rbp)
	jmp	.L103
.L87:
	movl	-76(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	dist(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	-64(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	leaq	dist(%rip), %rdx
	movl	(%rcx,%rdx), %ecx
	movl	-60(%rbp), %edx
	addl	%ecx, %edx
	cmpl	%edx, %eax
	jle	.L104
	movq	$29, -48(%rbp)
	jmp	.L103
.L104:
	movq	$11, -48(%rbp)
	jmp	.L103
.L99:
	cmpl	$25, -76(%rbp)
	jg	.L106
	movq	$1, -48(%rbp)
	jmp	.L103
.L106:
	movq	$0, -48(%rbp)
	jmp	.L103
.L84:
	cmpl	$25, -76(%rbp)
	jg	.L108
	movq	$10, -48(%rbp)
	jmp	.L103
.L108:
	movq	$17, -48(%rbp)
	jmp	.L103
.L83:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$14, -48(%rbp)
	jmp	.L103
.L96:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	getMinElementFromMinHeap
	movq	%rax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, -64(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	extractMinElementFromMinHeap
	movl	$0, -76(%rbp)
	movq	$4, -48(%rbp)
	jmp	.L103
.L101:
	movl	-76(%rbp), %eax
	cmpl	-64(%rbp), %eax
	je	.L111
	movq	$19, -48(%rbp)
	jmp	.L103
.L111:
	movq	$11, -48(%rbp)
	jmp	.L103
.L93:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-78(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-77(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movzbl	-78(%rbp), %eax
	movsbl	%al, %eax
	subl	$65, %eax
	movl	%eax, -72(%rbp)
	movzbl	-77(%rbp), %eax
	movsbl	%al, %eax
	subl	$65, %eax
	movl	%eax, -68(%rbp)
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	movl	$26, %esi
	movq	%rax, %rdi
	call	initializeMinHeap
	movl	$0, -76(%rbp)
	movq	$30, -48(%rbp)
	jmp	.L103
.L89:
	movl	-68(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	dist(%rip), %rax
	movl	(%rdx,%rax), %eax
	cmpl	$2147483647, %eax
	jne	.L113
	movq	$18, -48(%rbp)
	jmp	.L103
.L113:
	movq	$33, -48(%rbp)
	jmp	.L103
.L97:
	addl	$1, -76(%rbp)
	movq	$4, -48(%rbp)
	jmp	.L103
.L90:
	movl	-76(%rbp), %eax
	movslq	%eax, %rcx
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rdx
	leaq	adjacencyMatrix(%rip), %rax
	movl	(%rdx,%rax), %eax
	cmpl	$-1, %eax
	je	.L115
	movq	$22, -48(%rbp)
	jmp	.L103
.L115:
	movq	$11, -48(%rbp)
	jmp	.L103
.L92:
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	parentArr(%rip), %rax
	movl	$-1, (%rdx,%rax)
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	dist(%rip), %rax
	movl	$0, (%rdx,%rax)
	movl	-72(%rbp), %eax
	movl	%eax, -32(%rbp)
	movl	$0, -28(%rbp)
	movq	-32(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	insertElementToMinHeap
	movq	$0, -48(%rbp)
	jmp	.L103
.L86:
	movl	isFileRead(%rip), %eax
	testl	%eax, %eax
	jne	.L117
	movq	$31, -48(%rbp)
	jmp	.L103
.L117:
	movq	$16, -48(%rbp)
	jmp	.L103
.L88:
	movl	-76(%rbp), %eax
	movslq	%eax, %rcx
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rdx
	leaq	adjacencyMatrix(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -60(%rbp)
	movq	$25, -48(%rbp)
	jmp	.L103
.L81:
	movzbl	-77(%rbp), %eax
	movsbl	%al, %edx
	movzbl	-78(%rbp), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movzbl	-78(%rbp), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-68(%rbp), %eax
	movl	%eax, %edi
	call	printPath
	movl	-68(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	dist(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -48(%rbp)
	jmp	.L103
.L98:
	movl	-76(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	dist(%rip), %rax
	movl	$2147483647, (%rdx,%rax)
	addl	$1, -76(%rbp)
	movq	$30, -48(%rbp)
	jmp	.L103
.L102:
	movq	-56(%rbp), %rax
	movl	12(%rax), %eax
	testl	%eax, %eax
	jle	.L119
	movq	$12, -48(%rbp)
	jmp	.L103
.L119:
	movq	$21, -48(%rbp)
	jmp	.L103
.L85:
	movl	-64(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	dist(%rip), %rax
	movl	(%rdx,%rax), %edx
	movl	-60(%rbp), %eax
	leal	(%rdx,%rax), %ecx
	movl	-76(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	dist(%rip), %rax
	movl	%ecx, (%rdx,%rax)
	movl	-76(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	parentArr(%rip), %rdx
	movl	-64(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movl	-76(%rbp), %eax
	movl	%eax, -16(%rbp)
	movl	-76(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	dist(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -12(%rbp)
	movq	-16(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	insertElementToMinHeap
	movq	$11, -48(%rbp)
	jmp	.L103
.L100:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$13, -48(%rbp)
	jmp	.L103
.L124:
	nop
.L103:
	jmp	.L121
.L125:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L123
	call	__stack_chk_fail@PLT
.L123:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	shortestPath, .-shortestPath
	.globl	getMinElementFromMinHeap
	.type	getMinElementFromMinHeap, @function
getMinElementFromMinHeap:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L129:
	cmpq	$0, -8(%rbp)
	jne	.L132
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	(%rax), %rax
	jmp	.L131
.L132:
	nop
	jmp	.L129
.L131:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	getMinElementFromMinHeap, .-getMinElementFromMinHeap
	.globl	leftChildOfHeapNode
	.type	leftChildOfHeapNode, @function
leftChildOfHeapNode:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L136:
	cmpq	$0, -8(%rbp)
	jne	.L139
	movl	-20(%rbp), %eax
	addl	%eax, %eax
	addl	$1, %eax
	jmp	.L138
.L139:
	nop
	jmp	.L136
.L138:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	leftChildOfHeapNode, .-leftChildOfHeapNode
	.section	.rodata
.LC18:
	.string	"Good bye!\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movl	$0, -32(%rbp)
	jmp	.L141
.L142:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	parentArr(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -32(%rbp)
.L141:
	cmpl	$25, -32(%rbp)
	jle	.L142
	nop
.L143:
	movl	$0, -28(%rbp)
	jmp	.L144
.L145:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	dist(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -28(%rbp)
.L144:
	cmpl	$25, -28(%rbp)
	jle	.L145
	nop
.L146:
	movl	$0, -24(%rbp)
	jmp	.L147
.L148:
	movl	-24(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	isExists(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -24(%rbp)
.L147:
	cmpl	$25, -24(%rbp)
	jle	.L148
	nop
.L149:
	movl	$0, adjacencyMatrix(%rip)
	movl	$0, 4+adjacencyMatrix(%rip)
	movl	$0, 8+adjacencyMatrix(%rip)
	movl	$0, 12+adjacencyMatrix(%rip)
	movl	$0, 16+adjacencyMatrix(%rip)
	movl	$0, 20+adjacencyMatrix(%rip)
	movl	$0, 24+adjacencyMatrix(%rip)
	movl	$0, 28+adjacencyMatrix(%rip)
	movl	$0, 32+adjacencyMatrix(%rip)
	movl	$0, 36+adjacencyMatrix(%rip)
	movl	$0, 40+adjacencyMatrix(%rip)
	movl	$0, 44+adjacencyMatrix(%rip)
	movl	$0, 48+adjacencyMatrix(%rip)
	movl	$0, 52+adjacencyMatrix(%rip)
	movl	$0, 56+adjacencyMatrix(%rip)
	movl	$0, 60+adjacencyMatrix(%rip)
	movl	$0, 64+adjacencyMatrix(%rip)
	movl	$0, 68+adjacencyMatrix(%rip)
	movl	$0, 72+adjacencyMatrix(%rip)
	movl	$0, 76+adjacencyMatrix(%rip)
	movl	$0, 80+adjacencyMatrix(%rip)
	movl	$0, 84+adjacencyMatrix(%rip)
	movl	$0, 88+adjacencyMatrix(%rip)
	movl	$0, 92+adjacencyMatrix(%rip)
	movl	$0, 96+adjacencyMatrix(%rip)
	movl	$0, 100+adjacencyMatrix(%rip)
	movl	$0, 104+adjacencyMatrix(%rip)
	movl	$0, 108+adjacencyMatrix(%rip)
	movl	$0, 112+adjacencyMatrix(%rip)
	movl	$0, 116+adjacencyMatrix(%rip)
	movl	$0, 120+adjacencyMatrix(%rip)
	movl	$0, 124+adjacencyMatrix(%rip)
	movl	$0, 128+adjacencyMatrix(%rip)
	movl	$0, 132+adjacencyMatrix(%rip)
	movl	$0, 136+adjacencyMatrix(%rip)
	movl	$0, 140+adjacencyMatrix(%rip)
	movl	$0, 144+adjacencyMatrix(%rip)
	movl	$0, 148+adjacencyMatrix(%rip)
	movl	$0, 152+adjacencyMatrix(%rip)
	movl	$0, 156+adjacencyMatrix(%rip)
	movl	$0, 160+adjacencyMatrix(%rip)
	movl	$0, 164+adjacencyMatrix(%rip)
	movl	$0, 168+adjacencyMatrix(%rip)
	movl	$0, 172+adjacencyMatrix(%rip)
	movl	$0, 176+adjacencyMatrix(%rip)
	movl	$0, 180+adjacencyMatrix(%rip)
	movl	$0, 184+adjacencyMatrix(%rip)
	movl	$0, 188+adjacencyMatrix(%rip)
	movl	$0, 192+adjacencyMatrix(%rip)
	movl	$0, 196+adjacencyMatrix(%rip)
	movl	$0, 200+adjacencyMatrix(%rip)
	movl	$0, 204+adjacencyMatrix(%rip)
	movl	$0, 208+adjacencyMatrix(%rip)
	movl	$0, 212+adjacencyMatrix(%rip)
	movl	$0, 216+adjacencyMatrix(%rip)
	movl	$0, 220+adjacencyMatrix(%rip)
	movl	$0, 224+adjacencyMatrix(%rip)
	movl	$0, 228+adjacencyMatrix(%rip)
	movl	$0, 232+adjacencyMatrix(%rip)
	movl	$0, 236+adjacencyMatrix(%rip)
	movl	$0, 240+adjacencyMatrix(%rip)
	movl	$0, 244+adjacencyMatrix(%rip)
	movl	$0, 248+adjacencyMatrix(%rip)
	movl	$0, 252+adjacencyMatrix(%rip)
	movl	$0, 256+adjacencyMatrix(%rip)
	movl	$0, 260+adjacencyMatrix(%rip)
	movl	$0, 264+adjacencyMatrix(%rip)
	movl	$0, 268+adjacencyMatrix(%rip)
	movl	$0, 272+adjacencyMatrix(%rip)
	movl	$0, 276+adjacencyMatrix(%rip)
	movl	$0, 280+adjacencyMatrix(%rip)
	movl	$0, 284+adjacencyMatrix(%rip)
	movl	$0, 288+adjacencyMatrix(%rip)
	movl	$0, 292+adjacencyMatrix(%rip)
	movl	$0, 296+adjacencyMatrix(%rip)
	movl	$0, 300+adjacencyMatrix(%rip)
	movl	$0, 304+adjacencyMatrix(%rip)
	movl	$0, 308+adjacencyMatrix(%rip)
	movl	$0, 312+adjacencyMatrix(%rip)
	movl	$0, 316+adjacencyMatrix(%rip)
	movl	$0, 320+adjacencyMatrix(%rip)
	movl	$0, 324+adjacencyMatrix(%rip)
	movl	$0, 328+adjacencyMatrix(%rip)
	movl	$0, 332+adjacencyMatrix(%rip)
	movl	$0, 336+adjacencyMatrix(%rip)
	movl	$0, 340+adjacencyMatrix(%rip)
	movl	$0, 344+adjacencyMatrix(%rip)
	movl	$0, 348+adjacencyMatrix(%rip)
	movl	$0, 352+adjacencyMatrix(%rip)
	movl	$0, 356+adjacencyMatrix(%rip)
	movl	$0, 360+adjacencyMatrix(%rip)
	movl	$0, 364+adjacencyMatrix(%rip)
	movl	$0, 368+adjacencyMatrix(%rip)
	movl	$0, 372+adjacencyMatrix(%rip)
	movl	$0, 376+adjacencyMatrix(%rip)
	movl	$0, 380+adjacencyMatrix(%rip)
	movl	$0, 384+adjacencyMatrix(%rip)
	movl	$0, 388+adjacencyMatrix(%rip)
	movl	$0, 392+adjacencyMatrix(%rip)
	movl	$0, 396+adjacencyMatrix(%rip)
	movl	$0, 400+adjacencyMatrix(%rip)
	movl	$0, 404+adjacencyMatrix(%rip)
	movl	$0, 408+adjacencyMatrix(%rip)
	movl	$0, 412+adjacencyMatrix(%rip)
	movl	$0, 416+adjacencyMatrix(%rip)
	movl	$0, 420+adjacencyMatrix(%rip)
	movl	$0, 424+adjacencyMatrix(%rip)
	movl	$0, 428+adjacencyMatrix(%rip)
	movl	$0, 432+adjacencyMatrix(%rip)
	movl	$0, 436+adjacencyMatrix(%rip)
	movl	$0, 440+adjacencyMatrix(%rip)
	movl	$0, 444+adjacencyMatrix(%rip)
	movl	$0, 448+adjacencyMatrix(%rip)
	movl	$0, 452+adjacencyMatrix(%rip)
	movl	$0, 456+adjacencyMatrix(%rip)
	movl	$0, 460+adjacencyMatrix(%rip)
	movl	$0, 464+adjacencyMatrix(%rip)
	movl	$0, 468+adjacencyMatrix(%rip)
	movl	$0, 472+adjacencyMatrix(%rip)
	movl	$0, 476+adjacencyMatrix(%rip)
	movl	$0, 480+adjacencyMatrix(%rip)
	movl	$0, 484+adjacencyMatrix(%rip)
	movl	$0, 488+adjacencyMatrix(%rip)
	movl	$0, 492+adjacencyMatrix(%rip)
	movl	$0, 496+adjacencyMatrix(%rip)
	movl	$0, 500+adjacencyMatrix(%rip)
	movl	$0, 504+adjacencyMatrix(%rip)
	movl	$0, 508+adjacencyMatrix(%rip)
	movl	$0, 512+adjacencyMatrix(%rip)
	movl	$0, 516+adjacencyMatrix(%rip)
	movl	$0, 520+adjacencyMatrix(%rip)
	movl	$0, 524+adjacencyMatrix(%rip)
	movl	$0, 528+adjacencyMatrix(%rip)
	movl	$0, 532+adjacencyMatrix(%rip)
	movl	$0, 536+adjacencyMatrix(%rip)
	movl	$0, 540+adjacencyMatrix(%rip)
	movl	$0, 544+adjacencyMatrix(%rip)
	movl	$0, 548+adjacencyMatrix(%rip)
	movl	$0, 552+adjacencyMatrix(%rip)
	movl	$0, 556+adjacencyMatrix(%rip)
	movl	$0, 560+adjacencyMatrix(%rip)
	movl	$0, 564+adjacencyMatrix(%rip)
	movl	$0, 568+adjacencyMatrix(%rip)
	movl	$0, 572+adjacencyMatrix(%rip)
	movl	$0, 576+adjacencyMatrix(%rip)
	movl	$0, 580+adjacencyMatrix(%rip)
	movl	$0, 584+adjacencyMatrix(%rip)
	movl	$0, 588+adjacencyMatrix(%rip)
	movl	$0, 592+adjacencyMatrix(%rip)
	movl	$0, 596+adjacencyMatrix(%rip)
	movl	$0, 600+adjacencyMatrix(%rip)
	movl	$0, 604+adjacencyMatrix(%rip)
	movl	$0, 608+adjacencyMatrix(%rip)
	movl	$0, 612+adjacencyMatrix(%rip)
	movl	$0, 616+adjacencyMatrix(%rip)
	movl	$0, 620+adjacencyMatrix(%rip)
	movl	$0, 624+adjacencyMatrix(%rip)
	movl	$0, 628+adjacencyMatrix(%rip)
	movl	$0, 632+adjacencyMatrix(%rip)
	movl	$0, 636+adjacencyMatrix(%rip)
	movl	$0, 640+adjacencyMatrix(%rip)
	movl	$0, 644+adjacencyMatrix(%rip)
	movl	$0, 648+adjacencyMatrix(%rip)
	movl	$0, 652+adjacencyMatrix(%rip)
	movl	$0, 656+adjacencyMatrix(%rip)
	movl	$0, 660+adjacencyMatrix(%rip)
	movl	$0, 664+adjacencyMatrix(%rip)
	movl	$0, 668+adjacencyMatrix(%rip)
	movl	$0, 672+adjacencyMatrix(%rip)
	movl	$0, 676+adjacencyMatrix(%rip)
	movl	$0, 680+adjacencyMatrix(%rip)
	movl	$0, 684+adjacencyMatrix(%rip)
	movl	$0, 688+adjacencyMatrix(%rip)
	movl	$0, 692+adjacencyMatrix(%rip)
	movl	$0, 696+adjacencyMatrix(%rip)
	movl	$0, 700+adjacencyMatrix(%rip)
	movl	$0, 704+adjacencyMatrix(%rip)
	movl	$0, 708+adjacencyMatrix(%rip)
	movl	$0, 712+adjacencyMatrix(%rip)
	movl	$0, 716+adjacencyMatrix(%rip)
	movl	$0, 720+adjacencyMatrix(%rip)
	movl	$0, 724+adjacencyMatrix(%rip)
	movl	$0, 728+adjacencyMatrix(%rip)
	movl	$0, 732+adjacencyMatrix(%rip)
	movl	$0, 736+adjacencyMatrix(%rip)
	movl	$0, 740+adjacencyMatrix(%rip)
	movl	$0, 744+adjacencyMatrix(%rip)
	movl	$0, 748+adjacencyMatrix(%rip)
	movl	$0, 752+adjacencyMatrix(%rip)
	movl	$0, 756+adjacencyMatrix(%rip)
	movl	$0, 760+adjacencyMatrix(%rip)
	movl	$0, 764+adjacencyMatrix(%rip)
	movl	$0, 768+adjacencyMatrix(%rip)
	movl	$0, 772+adjacencyMatrix(%rip)
	movl	$0, 776+adjacencyMatrix(%rip)
	movl	$0, 780+adjacencyMatrix(%rip)
	movl	$0, 784+adjacencyMatrix(%rip)
	movl	$0, 788+adjacencyMatrix(%rip)
	movl	$0, 792+adjacencyMatrix(%rip)
	movl	$0, 796+adjacencyMatrix(%rip)
	movl	$0, 800+adjacencyMatrix(%rip)
	movl	$0, 804+adjacencyMatrix(%rip)
	movl	$0, 808+adjacencyMatrix(%rip)
	movl	$0, 812+adjacencyMatrix(%rip)
	movl	$0, 816+adjacencyMatrix(%rip)
	movl	$0, 820+adjacencyMatrix(%rip)
	movl	$0, 824+adjacencyMatrix(%rip)
	movl	$0, 828+adjacencyMatrix(%rip)
	movl	$0, 832+adjacencyMatrix(%rip)
	movl	$0, 836+adjacencyMatrix(%rip)
	movl	$0, 840+adjacencyMatrix(%rip)
	movl	$0, 844+adjacencyMatrix(%rip)
	movl	$0, 848+adjacencyMatrix(%rip)
	movl	$0, 852+adjacencyMatrix(%rip)
	movl	$0, 856+adjacencyMatrix(%rip)
	movl	$0, 860+adjacencyMatrix(%rip)
	movl	$0, 864+adjacencyMatrix(%rip)
	movl	$0, 868+adjacencyMatrix(%rip)
	movl	$0, 872+adjacencyMatrix(%rip)
	movl	$0, 876+adjacencyMatrix(%rip)
	movl	$0, 880+adjacencyMatrix(%rip)
	movl	$0, 884+adjacencyMatrix(%rip)
	movl	$0, 888+adjacencyMatrix(%rip)
	movl	$0, 892+adjacencyMatrix(%rip)
	movl	$0, 896+adjacencyMatrix(%rip)
	movl	$0, 900+adjacencyMatrix(%rip)
	movl	$0, 904+adjacencyMatrix(%rip)
	movl	$0, 908+adjacencyMatrix(%rip)
	movl	$0, 912+adjacencyMatrix(%rip)
	movl	$0, 916+adjacencyMatrix(%rip)
	movl	$0, 920+adjacencyMatrix(%rip)
	movl	$0, 924+adjacencyMatrix(%rip)
	movl	$0, 928+adjacencyMatrix(%rip)
	movl	$0, 932+adjacencyMatrix(%rip)
	movl	$0, 936+adjacencyMatrix(%rip)
	movl	$0, 940+adjacencyMatrix(%rip)
	movl	$0, 944+adjacencyMatrix(%rip)
	movl	$0, 948+adjacencyMatrix(%rip)
	movl	$0, 952+adjacencyMatrix(%rip)
	movl	$0, 956+adjacencyMatrix(%rip)
	movl	$0, 960+adjacencyMatrix(%rip)
	movl	$0, 964+adjacencyMatrix(%rip)
	movl	$0, 968+adjacencyMatrix(%rip)
	movl	$0, 972+adjacencyMatrix(%rip)
	movl	$0, 976+adjacencyMatrix(%rip)
	movl	$0, 980+adjacencyMatrix(%rip)
	movl	$0, 984+adjacencyMatrix(%rip)
	movl	$0, 988+adjacencyMatrix(%rip)
	movl	$0, 992+adjacencyMatrix(%rip)
	movl	$0, 996+adjacencyMatrix(%rip)
	movl	$0, 1000+adjacencyMatrix(%rip)
	movl	$0, 1004+adjacencyMatrix(%rip)
	movl	$0, 1008+adjacencyMatrix(%rip)
	movl	$0, 1012+adjacencyMatrix(%rip)
	movl	$0, 1016+adjacencyMatrix(%rip)
	movl	$0, 1020+adjacencyMatrix(%rip)
	movl	$0, 1024+adjacencyMatrix(%rip)
	movl	$0, 1028+adjacencyMatrix(%rip)
	movl	$0, 1032+adjacencyMatrix(%rip)
	movl	$0, 1036+adjacencyMatrix(%rip)
	movl	$0, 1040+adjacencyMatrix(%rip)
	movl	$0, 1044+adjacencyMatrix(%rip)
	movl	$0, 1048+adjacencyMatrix(%rip)
	movl	$0, 1052+adjacencyMatrix(%rip)
	movl	$0, 1056+adjacencyMatrix(%rip)
	movl	$0, 1060+adjacencyMatrix(%rip)
	movl	$0, 1064+adjacencyMatrix(%rip)
	movl	$0, 1068+adjacencyMatrix(%rip)
	movl	$0, 1072+adjacencyMatrix(%rip)
	movl	$0, 1076+adjacencyMatrix(%rip)
	movl	$0, 1080+adjacencyMatrix(%rip)
	movl	$0, 1084+adjacencyMatrix(%rip)
	movl	$0, 1088+adjacencyMatrix(%rip)
	movl	$0, 1092+adjacencyMatrix(%rip)
	movl	$0, 1096+adjacencyMatrix(%rip)
	movl	$0, 1100+adjacencyMatrix(%rip)
	movl	$0, 1104+adjacencyMatrix(%rip)
	movl	$0, 1108+adjacencyMatrix(%rip)
	movl	$0, 1112+adjacencyMatrix(%rip)
	movl	$0, 1116+adjacencyMatrix(%rip)
	movl	$0, 1120+adjacencyMatrix(%rip)
	movl	$0, 1124+adjacencyMatrix(%rip)
	movl	$0, 1128+adjacencyMatrix(%rip)
	movl	$0, 1132+adjacencyMatrix(%rip)
	movl	$0, 1136+adjacencyMatrix(%rip)
	movl	$0, 1140+adjacencyMatrix(%rip)
	movl	$0, 1144+adjacencyMatrix(%rip)
	movl	$0, 1148+adjacencyMatrix(%rip)
	movl	$0, 1152+adjacencyMatrix(%rip)
	movl	$0, 1156+adjacencyMatrix(%rip)
	movl	$0, 1160+adjacencyMatrix(%rip)
	movl	$0, 1164+adjacencyMatrix(%rip)
	movl	$0, 1168+adjacencyMatrix(%rip)
	movl	$0, 1172+adjacencyMatrix(%rip)
	movl	$0, 1176+adjacencyMatrix(%rip)
	movl	$0, 1180+adjacencyMatrix(%rip)
	movl	$0, 1184+adjacencyMatrix(%rip)
	movl	$0, 1188+adjacencyMatrix(%rip)
	movl	$0, 1192+adjacencyMatrix(%rip)
	movl	$0, 1196+adjacencyMatrix(%rip)
	movl	$0, 1200+adjacencyMatrix(%rip)
	movl	$0, 1204+adjacencyMatrix(%rip)
	movl	$0, 1208+adjacencyMatrix(%rip)
	movl	$0, 1212+adjacencyMatrix(%rip)
	movl	$0, 1216+adjacencyMatrix(%rip)
	movl	$0, 1220+adjacencyMatrix(%rip)
	movl	$0, 1224+adjacencyMatrix(%rip)
	movl	$0, 1228+adjacencyMatrix(%rip)
	movl	$0, 1232+adjacencyMatrix(%rip)
	movl	$0, 1236+adjacencyMatrix(%rip)
	movl	$0, 1240+adjacencyMatrix(%rip)
	movl	$0, 1244+adjacencyMatrix(%rip)
	movl	$0, 1248+adjacencyMatrix(%rip)
	movl	$0, 1252+adjacencyMatrix(%rip)
	movl	$0, 1256+adjacencyMatrix(%rip)
	movl	$0, 1260+adjacencyMatrix(%rip)
	movl	$0, 1264+adjacencyMatrix(%rip)
	movl	$0, 1268+adjacencyMatrix(%rip)
	movl	$0, 1272+adjacencyMatrix(%rip)
	movl	$0, 1276+adjacencyMatrix(%rip)
	movl	$0, 1280+adjacencyMatrix(%rip)
	movl	$0, 1284+adjacencyMatrix(%rip)
	movl	$0, 1288+adjacencyMatrix(%rip)
	movl	$0, 1292+adjacencyMatrix(%rip)
	movl	$0, 1296+adjacencyMatrix(%rip)
	movl	$0, 1300+adjacencyMatrix(%rip)
	movl	$0, 1304+adjacencyMatrix(%rip)
	movl	$0, 1308+adjacencyMatrix(%rip)
	movl	$0, 1312+adjacencyMatrix(%rip)
	movl	$0, 1316+adjacencyMatrix(%rip)
	movl	$0, 1320+adjacencyMatrix(%rip)
	movl	$0, 1324+adjacencyMatrix(%rip)
	movl	$0, 1328+adjacencyMatrix(%rip)
	movl	$0, 1332+adjacencyMatrix(%rip)
	movl	$0, 1336+adjacencyMatrix(%rip)
	movl	$0, 1340+adjacencyMatrix(%rip)
	movl	$0, 1344+adjacencyMatrix(%rip)
	movl	$0, 1348+adjacencyMatrix(%rip)
	movl	$0, 1352+adjacencyMatrix(%rip)
	movl	$0, 1356+adjacencyMatrix(%rip)
	movl	$0, 1360+adjacencyMatrix(%rip)
	movl	$0, 1364+adjacencyMatrix(%rip)
	movl	$0, 1368+adjacencyMatrix(%rip)
	movl	$0, 1372+adjacencyMatrix(%rip)
	movl	$0, 1376+adjacencyMatrix(%rip)
	movl	$0, 1380+adjacencyMatrix(%rip)
	movl	$0, 1384+adjacencyMatrix(%rip)
	movl	$0, 1388+adjacencyMatrix(%rip)
	movl	$0, 1392+adjacencyMatrix(%rip)
	movl	$0, 1396+adjacencyMatrix(%rip)
	movl	$0, 1400+adjacencyMatrix(%rip)
	movl	$0, 1404+adjacencyMatrix(%rip)
	movl	$0, 1408+adjacencyMatrix(%rip)
	movl	$0, 1412+adjacencyMatrix(%rip)
	movl	$0, 1416+adjacencyMatrix(%rip)
	movl	$0, 1420+adjacencyMatrix(%rip)
	movl	$0, 1424+adjacencyMatrix(%rip)
	movl	$0, 1428+adjacencyMatrix(%rip)
	movl	$0, 1432+adjacencyMatrix(%rip)
	movl	$0, 1436+adjacencyMatrix(%rip)
	movl	$0, 1440+adjacencyMatrix(%rip)
	movl	$0, 1444+adjacencyMatrix(%rip)
	movl	$0, 1448+adjacencyMatrix(%rip)
	movl	$0, 1452+adjacencyMatrix(%rip)
	movl	$0, 1456+adjacencyMatrix(%rip)
	movl	$0, 1460+adjacencyMatrix(%rip)
	movl	$0, 1464+adjacencyMatrix(%rip)
	movl	$0, 1468+adjacencyMatrix(%rip)
	movl	$0, 1472+adjacencyMatrix(%rip)
	movl	$0, 1476+adjacencyMatrix(%rip)
	movl	$0, 1480+adjacencyMatrix(%rip)
	movl	$0, 1484+adjacencyMatrix(%rip)
	movl	$0, 1488+adjacencyMatrix(%rip)
	movl	$0, 1492+adjacencyMatrix(%rip)
	movl	$0, 1496+adjacencyMatrix(%rip)
	movl	$0, 1500+adjacencyMatrix(%rip)
	movl	$0, 1504+adjacencyMatrix(%rip)
	movl	$0, 1508+adjacencyMatrix(%rip)
	movl	$0, 1512+adjacencyMatrix(%rip)
	movl	$0, 1516+adjacencyMatrix(%rip)
	movl	$0, 1520+adjacencyMatrix(%rip)
	movl	$0, 1524+adjacencyMatrix(%rip)
	movl	$0, 1528+adjacencyMatrix(%rip)
	movl	$0, 1532+adjacencyMatrix(%rip)
	movl	$0, 1536+adjacencyMatrix(%rip)
	movl	$0, 1540+adjacencyMatrix(%rip)
	movl	$0, 1544+adjacencyMatrix(%rip)
	movl	$0, 1548+adjacencyMatrix(%rip)
	movl	$0, 1552+adjacencyMatrix(%rip)
	movl	$0, 1556+adjacencyMatrix(%rip)
	movl	$0, 1560+adjacencyMatrix(%rip)
	movl	$0, 1564+adjacencyMatrix(%rip)
	movl	$0, 1568+adjacencyMatrix(%rip)
	movl	$0, 1572+adjacencyMatrix(%rip)
	movl	$0, 1576+adjacencyMatrix(%rip)
	movl	$0, 1580+adjacencyMatrix(%rip)
	movl	$0, 1584+adjacencyMatrix(%rip)
	movl	$0, 1588+adjacencyMatrix(%rip)
	movl	$0, 1592+adjacencyMatrix(%rip)
	movl	$0, 1596+adjacencyMatrix(%rip)
	movl	$0, 1600+adjacencyMatrix(%rip)
	movl	$0, 1604+adjacencyMatrix(%rip)
	movl	$0, 1608+adjacencyMatrix(%rip)
	movl	$0, 1612+adjacencyMatrix(%rip)
	movl	$0, 1616+adjacencyMatrix(%rip)
	movl	$0, 1620+adjacencyMatrix(%rip)
	movl	$0, 1624+adjacencyMatrix(%rip)
	movl	$0, 1628+adjacencyMatrix(%rip)
	movl	$0, 1632+adjacencyMatrix(%rip)
	movl	$0, 1636+adjacencyMatrix(%rip)
	movl	$0, 1640+adjacencyMatrix(%rip)
	movl	$0, 1644+adjacencyMatrix(%rip)
	movl	$0, 1648+adjacencyMatrix(%rip)
	movl	$0, 1652+adjacencyMatrix(%rip)
	movl	$0, 1656+adjacencyMatrix(%rip)
	movl	$0, 1660+adjacencyMatrix(%rip)
	movl	$0, 1664+adjacencyMatrix(%rip)
	movl	$0, 1668+adjacencyMatrix(%rip)
	movl	$0, 1672+adjacencyMatrix(%rip)
	movl	$0, 1676+adjacencyMatrix(%rip)
	movl	$0, 1680+adjacencyMatrix(%rip)
	movl	$0, 1684+adjacencyMatrix(%rip)
	movl	$0, 1688+adjacencyMatrix(%rip)
	movl	$0, 1692+adjacencyMatrix(%rip)
	movl	$0, 1696+adjacencyMatrix(%rip)
	movl	$0, 1700+adjacencyMatrix(%rip)
	movl	$0, 1704+adjacencyMatrix(%rip)
	movl	$0, 1708+adjacencyMatrix(%rip)
	movl	$0, 1712+adjacencyMatrix(%rip)
	movl	$0, 1716+adjacencyMatrix(%rip)
	movl	$0, 1720+adjacencyMatrix(%rip)
	movl	$0, 1724+adjacencyMatrix(%rip)
	movl	$0, 1728+adjacencyMatrix(%rip)
	movl	$0, 1732+adjacencyMatrix(%rip)
	movl	$0, 1736+adjacencyMatrix(%rip)
	movl	$0, 1740+adjacencyMatrix(%rip)
	movl	$0, 1744+adjacencyMatrix(%rip)
	movl	$0, 1748+adjacencyMatrix(%rip)
	movl	$0, 1752+adjacencyMatrix(%rip)
	movl	$0, 1756+adjacencyMatrix(%rip)
	movl	$0, 1760+adjacencyMatrix(%rip)
	movl	$0, 1764+adjacencyMatrix(%rip)
	movl	$0, 1768+adjacencyMatrix(%rip)
	movl	$0, 1772+adjacencyMatrix(%rip)
	movl	$0, 1776+adjacencyMatrix(%rip)
	movl	$0, 1780+adjacencyMatrix(%rip)
	movl	$0, 1784+adjacencyMatrix(%rip)
	movl	$0, 1788+adjacencyMatrix(%rip)
	movl	$0, 1792+adjacencyMatrix(%rip)
	movl	$0, 1796+adjacencyMatrix(%rip)
	movl	$0, 1800+adjacencyMatrix(%rip)
	movl	$0, 1804+adjacencyMatrix(%rip)
	movl	$0, 1808+adjacencyMatrix(%rip)
	movl	$0, 1812+adjacencyMatrix(%rip)
	movl	$0, 1816+adjacencyMatrix(%rip)
	movl	$0, 1820+adjacencyMatrix(%rip)
	movl	$0, 1824+adjacencyMatrix(%rip)
	movl	$0, 1828+adjacencyMatrix(%rip)
	movl	$0, 1832+adjacencyMatrix(%rip)
	movl	$0, 1836+adjacencyMatrix(%rip)
	movl	$0, 1840+adjacencyMatrix(%rip)
	movl	$0, 1844+adjacencyMatrix(%rip)
	movl	$0, 1848+adjacencyMatrix(%rip)
	movl	$0, 1852+adjacencyMatrix(%rip)
	movl	$0, 1856+adjacencyMatrix(%rip)
	movl	$0, 1860+adjacencyMatrix(%rip)
	movl	$0, 1864+adjacencyMatrix(%rip)
	movl	$0, 1868+adjacencyMatrix(%rip)
	movl	$0, 1872+adjacencyMatrix(%rip)
	movl	$0, 1876+adjacencyMatrix(%rip)
	movl	$0, 1880+adjacencyMatrix(%rip)
	movl	$0, 1884+adjacencyMatrix(%rip)
	movl	$0, 1888+adjacencyMatrix(%rip)
	movl	$0, 1892+adjacencyMatrix(%rip)
	movl	$0, 1896+adjacencyMatrix(%rip)
	movl	$0, 1900+adjacencyMatrix(%rip)
	movl	$0, 1904+adjacencyMatrix(%rip)
	movl	$0, 1908+adjacencyMatrix(%rip)
	movl	$0, 1912+adjacencyMatrix(%rip)
	movl	$0, 1916+adjacencyMatrix(%rip)
	movl	$0, 1920+adjacencyMatrix(%rip)
	movl	$0, 1924+adjacencyMatrix(%rip)
	movl	$0, 1928+adjacencyMatrix(%rip)
	movl	$0, 1932+adjacencyMatrix(%rip)
	movl	$0, 1936+adjacencyMatrix(%rip)
	movl	$0, 1940+adjacencyMatrix(%rip)
	movl	$0, 1944+adjacencyMatrix(%rip)
	movl	$0, 1948+adjacencyMatrix(%rip)
	movl	$0, 1952+adjacencyMatrix(%rip)
	movl	$0, 1956+adjacencyMatrix(%rip)
	movl	$0, 1960+adjacencyMatrix(%rip)
	movl	$0, 1964+adjacencyMatrix(%rip)
	movl	$0, 1968+adjacencyMatrix(%rip)
	movl	$0, 1972+adjacencyMatrix(%rip)
	movl	$0, 1976+adjacencyMatrix(%rip)
	movl	$0, 1980+adjacencyMatrix(%rip)
	movl	$0, 1984+adjacencyMatrix(%rip)
	movl	$0, 1988+adjacencyMatrix(%rip)
	movl	$0, 1992+adjacencyMatrix(%rip)
	movl	$0, 1996+adjacencyMatrix(%rip)
	movl	$0, 2000+adjacencyMatrix(%rip)
	movl	$0, 2004+adjacencyMatrix(%rip)
	movl	$0, 2008+adjacencyMatrix(%rip)
	movl	$0, 2012+adjacencyMatrix(%rip)
	movl	$0, 2016+adjacencyMatrix(%rip)
	movl	$0, 2020+adjacencyMatrix(%rip)
	movl	$0, 2024+adjacencyMatrix(%rip)
	movl	$0, 2028+adjacencyMatrix(%rip)
	movl	$0, 2032+adjacencyMatrix(%rip)
	movl	$0, 2036+adjacencyMatrix(%rip)
	movl	$0, 2040+adjacencyMatrix(%rip)
	movl	$0, 2044+adjacencyMatrix(%rip)
	movl	$0, 2048+adjacencyMatrix(%rip)
	movl	$0, 2052+adjacencyMatrix(%rip)
	movl	$0, 2056+adjacencyMatrix(%rip)
	movl	$0, 2060+adjacencyMatrix(%rip)
	movl	$0, 2064+adjacencyMatrix(%rip)
	movl	$0, 2068+adjacencyMatrix(%rip)
	movl	$0, 2072+adjacencyMatrix(%rip)
	movl	$0, 2076+adjacencyMatrix(%rip)
	movl	$0, 2080+adjacencyMatrix(%rip)
	movl	$0, 2084+adjacencyMatrix(%rip)
	movl	$0, 2088+adjacencyMatrix(%rip)
	movl	$0, 2092+adjacencyMatrix(%rip)
	movl	$0, 2096+adjacencyMatrix(%rip)
	movl	$0, 2100+adjacencyMatrix(%rip)
	movl	$0, 2104+adjacencyMatrix(%rip)
	movl	$0, 2108+adjacencyMatrix(%rip)
	movl	$0, 2112+adjacencyMatrix(%rip)
	movl	$0, 2116+adjacencyMatrix(%rip)
	movl	$0, 2120+adjacencyMatrix(%rip)
	movl	$0, 2124+adjacencyMatrix(%rip)
	movl	$0, 2128+adjacencyMatrix(%rip)
	movl	$0, 2132+adjacencyMatrix(%rip)
	movl	$0, 2136+adjacencyMatrix(%rip)
	movl	$0, 2140+adjacencyMatrix(%rip)
	movl	$0, 2144+adjacencyMatrix(%rip)
	movl	$0, 2148+adjacencyMatrix(%rip)
	movl	$0, 2152+adjacencyMatrix(%rip)
	movl	$0, 2156+adjacencyMatrix(%rip)
	movl	$0, 2160+adjacencyMatrix(%rip)
	movl	$0, 2164+adjacencyMatrix(%rip)
	movl	$0, 2168+adjacencyMatrix(%rip)
	movl	$0, 2172+adjacencyMatrix(%rip)
	movl	$0, 2176+adjacencyMatrix(%rip)
	movl	$0, 2180+adjacencyMatrix(%rip)
	movl	$0, 2184+adjacencyMatrix(%rip)
	movl	$0, 2188+adjacencyMatrix(%rip)
	movl	$0, 2192+adjacencyMatrix(%rip)
	movl	$0, 2196+adjacencyMatrix(%rip)
	movl	$0, 2200+adjacencyMatrix(%rip)
	movl	$0, 2204+adjacencyMatrix(%rip)
	movl	$0, 2208+adjacencyMatrix(%rip)
	movl	$0, 2212+adjacencyMatrix(%rip)
	movl	$0, 2216+adjacencyMatrix(%rip)
	movl	$0, 2220+adjacencyMatrix(%rip)
	movl	$0, 2224+adjacencyMatrix(%rip)
	movl	$0, 2228+adjacencyMatrix(%rip)
	movl	$0, 2232+adjacencyMatrix(%rip)
	movl	$0, 2236+adjacencyMatrix(%rip)
	movl	$0, 2240+adjacencyMatrix(%rip)
	movl	$0, 2244+adjacencyMatrix(%rip)
	movl	$0, 2248+adjacencyMatrix(%rip)
	movl	$0, 2252+adjacencyMatrix(%rip)
	movl	$0, 2256+adjacencyMatrix(%rip)
	movl	$0, 2260+adjacencyMatrix(%rip)
	movl	$0, 2264+adjacencyMatrix(%rip)
	movl	$0, 2268+adjacencyMatrix(%rip)
	movl	$0, 2272+adjacencyMatrix(%rip)
	movl	$0, 2276+adjacencyMatrix(%rip)
	movl	$0, 2280+adjacencyMatrix(%rip)
	movl	$0, 2284+adjacencyMatrix(%rip)
	movl	$0, 2288+adjacencyMatrix(%rip)
	movl	$0, 2292+adjacencyMatrix(%rip)
	movl	$0, 2296+adjacencyMatrix(%rip)
	movl	$0, 2300+adjacencyMatrix(%rip)
	movl	$0, 2304+adjacencyMatrix(%rip)
	movl	$0, 2308+adjacencyMatrix(%rip)
	movl	$0, 2312+adjacencyMatrix(%rip)
	movl	$0, 2316+adjacencyMatrix(%rip)
	movl	$0, 2320+adjacencyMatrix(%rip)
	movl	$0, 2324+adjacencyMatrix(%rip)
	movl	$0, 2328+adjacencyMatrix(%rip)
	movl	$0, 2332+adjacencyMatrix(%rip)
	movl	$0, 2336+adjacencyMatrix(%rip)
	movl	$0, 2340+adjacencyMatrix(%rip)
	movl	$0, 2344+adjacencyMatrix(%rip)
	movl	$0, 2348+adjacencyMatrix(%rip)
	movl	$0, 2352+adjacencyMatrix(%rip)
	movl	$0, 2356+adjacencyMatrix(%rip)
	movl	$0, 2360+adjacencyMatrix(%rip)
	movl	$0, 2364+adjacencyMatrix(%rip)
	movl	$0, 2368+adjacencyMatrix(%rip)
	movl	$0, 2372+adjacencyMatrix(%rip)
	movl	$0, 2376+adjacencyMatrix(%rip)
	movl	$0, 2380+adjacencyMatrix(%rip)
	movl	$0, 2384+adjacencyMatrix(%rip)
	movl	$0, 2388+adjacencyMatrix(%rip)
	movl	$0, 2392+adjacencyMatrix(%rip)
	movl	$0, 2396+adjacencyMatrix(%rip)
	movl	$0, 2400+adjacencyMatrix(%rip)
	movl	$0, 2404+adjacencyMatrix(%rip)
	movl	$0, 2408+adjacencyMatrix(%rip)
	movl	$0, 2412+adjacencyMatrix(%rip)
	movl	$0, 2416+adjacencyMatrix(%rip)
	movl	$0, 2420+adjacencyMatrix(%rip)
	movl	$0, 2424+adjacencyMatrix(%rip)
	movl	$0, 2428+adjacencyMatrix(%rip)
	movl	$0, 2432+adjacencyMatrix(%rip)
	movl	$0, 2436+adjacencyMatrix(%rip)
	movl	$0, 2440+adjacencyMatrix(%rip)
	movl	$0, 2444+adjacencyMatrix(%rip)
	movl	$0, 2448+adjacencyMatrix(%rip)
	movl	$0, 2452+adjacencyMatrix(%rip)
	movl	$0, 2456+adjacencyMatrix(%rip)
	movl	$0, 2460+adjacencyMatrix(%rip)
	movl	$0, 2464+adjacencyMatrix(%rip)
	movl	$0, 2468+adjacencyMatrix(%rip)
	movl	$0, 2472+adjacencyMatrix(%rip)
	movl	$0, 2476+adjacencyMatrix(%rip)
	movl	$0, 2480+adjacencyMatrix(%rip)
	movl	$0, 2484+adjacencyMatrix(%rip)
	movl	$0, 2488+adjacencyMatrix(%rip)
	movl	$0, 2492+adjacencyMatrix(%rip)
	movl	$0, 2496+adjacencyMatrix(%rip)
	movl	$0, 2500+adjacencyMatrix(%rip)
	movl	$0, 2504+adjacencyMatrix(%rip)
	movl	$0, 2508+adjacencyMatrix(%rip)
	movl	$0, 2512+adjacencyMatrix(%rip)
	movl	$0, 2516+adjacencyMatrix(%rip)
	movl	$0, 2520+adjacencyMatrix(%rip)
	movl	$0, 2524+adjacencyMatrix(%rip)
	movl	$0, 2528+adjacencyMatrix(%rip)
	movl	$0, 2532+adjacencyMatrix(%rip)
	movl	$0, 2536+adjacencyMatrix(%rip)
	movl	$0, 2540+adjacencyMatrix(%rip)
	movl	$0, 2544+adjacencyMatrix(%rip)
	movl	$0, 2548+adjacencyMatrix(%rip)
	movl	$0, 2552+adjacencyMatrix(%rip)
	movl	$0, 2556+adjacencyMatrix(%rip)
	movl	$0, 2560+adjacencyMatrix(%rip)
	movl	$0, 2564+adjacencyMatrix(%rip)
	movl	$0, 2568+adjacencyMatrix(%rip)
	movl	$0, 2572+adjacencyMatrix(%rip)
	movl	$0, 2576+adjacencyMatrix(%rip)
	movl	$0, 2580+adjacencyMatrix(%rip)
	movl	$0, 2584+adjacencyMatrix(%rip)
	movl	$0, 2588+adjacencyMatrix(%rip)
	movl	$0, 2592+adjacencyMatrix(%rip)
	movl	$0, 2596+adjacencyMatrix(%rip)
	movl	$0, 2600+adjacencyMatrix(%rip)
	movl	$0, 2604+adjacencyMatrix(%rip)
	movl	$0, 2608+adjacencyMatrix(%rip)
	movl	$0, 2612+adjacencyMatrix(%rip)
	movl	$0, 2616+adjacencyMatrix(%rip)
	movl	$0, 2620+adjacencyMatrix(%rip)
	movl	$0, 2624+adjacencyMatrix(%rip)
	movl	$0, 2628+adjacencyMatrix(%rip)
	movl	$0, 2632+adjacencyMatrix(%rip)
	movl	$0, 2636+adjacencyMatrix(%rip)
	movl	$0, 2640+adjacencyMatrix(%rip)
	movl	$0, 2644+adjacencyMatrix(%rip)
	movl	$0, 2648+adjacencyMatrix(%rip)
	movl	$0, 2652+adjacencyMatrix(%rip)
	movl	$0, 2656+adjacencyMatrix(%rip)
	movl	$0, 2660+adjacencyMatrix(%rip)
	movl	$0, 2664+adjacencyMatrix(%rip)
	movl	$0, 2668+adjacencyMatrix(%rip)
	movl	$0, 2672+adjacencyMatrix(%rip)
	movl	$0, 2676+adjacencyMatrix(%rip)
	movl	$0, 2680+adjacencyMatrix(%rip)
	movl	$0, 2684+adjacencyMatrix(%rip)
	movl	$0, 2688+adjacencyMatrix(%rip)
	movl	$0, 2692+adjacencyMatrix(%rip)
	movl	$0, 2696+adjacencyMatrix(%rip)
	movl	$0, 2700+adjacencyMatrix(%rip)
	nop
.L150:
	movl	$0, isFileRead(%rip)
	nop
.L151:
	movq	$0, _TIG_IZ_XXjf_envp(%rip)
	nop
.L152:
	movq	$0, _TIG_IZ_XXjf_argv(%rip)
	nop
.L153:
	movl	$0, _TIG_IZ_XXjf_argc(%rip)
	nop
	nop
.L154:
.L155:
#APP
# 1165 "kadwani_lab1b_3.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XXjf--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_XXjf_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_XXjf_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_XXjf_envp(%rip)
	nop
	movq	$4, -8(%rbp)
.L183:
	cmpq	$16, -8(%rbp)
	ja	.L185
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L158(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L158(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L158:
	.long	.L170-.L158
	.long	.L185-.L158
	.long	.L185-.L158
	.long	.L169-.L158
	.long	.L168-.L158
	.long	.L167-.L158
	.long	.L166-.L158
	.long	.L165-.L158
	.long	.L185-.L158
	.long	.L164-.L158
	.long	.L163-.L158
	.long	.L162-.L158
	.long	.L161-.L158
	.long	.L160-.L158
	.long	.L159-.L158
	.long	.L185-.L158
	.long	.L157-.L158
	.text
.L168:
	movq	$6, -8(%rbp)
	jmp	.L171
.L159:
	cmpl	$0, -20(%rbp)
	jne	.L172
	movq	$12, -8(%rbp)
	jmp	.L171
.L172:
	movq	$7, -8(%rbp)
	jmp	.L171
.L161:
	call	showMenuAndGetChoice
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L171
.L169:
	cmpl	$3, -16(%rbp)
	jne	.L174
	movq	$9, -8(%rbp)
	jmp	.L171
.L174:
	movq	$13, -8(%rbp)
	jmp	.L171
.L157:
	cmpl	$2, -16(%rbp)
	jne	.L176
	movq	$5, -8(%rbp)
	jmp	.L171
.L176:
	movq	$3, -8(%rbp)
	jmp	.L171
.L162:
	call	readInputFile
	movq	$14, -8(%rbp)
	jmp	.L171
.L164:
	call	shortestPath
	movq	$14, -8(%rbp)
	jmp	.L171
.L160:
	cmpl	$4, -16(%rbp)
	jne	.L178
	movq	$0, -8(%rbp)
	jmp	.L171
.L178:
	movq	$14, -8(%rbp)
	jmp	.L171
.L166:
	movl	$0, isFileRead(%rip)
	movl	$0, -20(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L171
.L167:
	call	showAdjacencyMatrix
	movq	$14, -8(%rbp)
	jmp	.L171
.L163:
	cmpl	$1, -16(%rbp)
	jne	.L180
	movq	$11, -8(%rbp)
	jmp	.L171
.L180:
	movq	$16, -8(%rbp)
	jmp	.L171
.L170:
	movl	$1, -20(%rbp)
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$14, -8(%rbp)
	jmp	.L171
.L165:
	movl	$0, %eax
	jmp	.L184
.L185:
	nop
.L171:
	jmp	.L183
.L184:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	heapify
	.type	heapify, @function
heapify:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	$0, -8(%rbp)
.L212:
	cmpq	$11, -8(%rbp)
	ja	.L213
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L189(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L189(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L189:
	.long	.L199-.L189
	.long	.L198-.L189
	.long	.L197-.L189
	.long	.L214-.L189
	.long	.L213-.L189
	.long	.L195-.L189
	.long	.L194-.L189
	.long	.L193-.L189
	.long	.L192-.L189
	.long	.L191-.L189
	.long	.L190-.L189
	.long	.L188-.L189
	.text
.L192:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-20(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rcx, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	swapMinHeapNodes
	movl	-20(%rbp), %edx
	movq	-40(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	heapify
	movq	$3, -8(%rbp)
	jmp	.L200
.L198:
	movl	-24(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L200
.L188:
	movl	-28(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L200
.L191:
	movl	-44(%rbp), %eax
	movl	%eax, %edi
	call	leftChildOfHeapNode
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -28(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, %edi
	call	rightChildOfHeapNode
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -24(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L200
.L194:
	movq	-40(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jge	.L202
	movq	$7, -8(%rbp)
	jmp	.L200
.L202:
	movq	$5, -8(%rbp)
	jmp	.L200
.L195:
	movq	-40(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	%eax, -24(%rbp)
	jge	.L204
	movq	$10, -8(%rbp)
	jmp	.L200
.L204:
	movq	$2, -8(%rbp)
	jmp	.L200
.L190:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-24(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movl	4(%rax), %edx
	movq	-40(%rbp), %rax
	movq	(%rax), %rcx
	movl	-20(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rcx, %rax
	movl	4(%rax), %eax
	cmpl	%eax, %edx
	jge	.L206
	movq	$1, -8(%rbp)
	jmp	.L200
.L206:
	movq	$2, -8(%rbp)
	jmp	.L200
.L199:
	movq	$9, -8(%rbp)
	jmp	.L200
.L193:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movl	4(%rax), %edx
	movq	-40(%rbp), %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rcx, %rax
	movl	4(%rax), %eax
	cmpl	%eax, %edx
	jge	.L208
	movq	$11, -8(%rbp)
	jmp	.L200
.L208:
	movq	$5, -8(%rbp)
	jmp	.L200
.L197:
	movl	-20(%rbp), %eax
	cmpl	-44(%rbp), %eax
	je	.L210
	movq	$8, -8(%rbp)
	jmp	.L200
.L210:
	movq	$3, -8(%rbp)
	jmp	.L200
.L213:
	nop
.L200:
	jmp	.L212
.L214:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	heapify, .-heapify
	.globl	swapMinHeapNodes
	.type	swapMinHeapNodes, @function
swapMinHeapNodes:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$1, -8(%rbp)
.L221:
	cmpq	$2, -8(%rbp)
	je	.L222
	cmpq	$2, -8(%rbp)
	ja	.L223
	cmpq	$0, -8(%rbp)
	je	.L218
	cmpq	$1, -8(%rbp)
	jne	.L223
	movq	$0, -8(%rbp)
	jmp	.L219
.L218:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -16(%rbp)
	movq	-24(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	-32(%rbp), %rax
	movl	(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	-32(%rbp), %rax
	movl	4(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, 4(%rax)
	movq	-32(%rbp), %rax
	movl	-16(%rbp), %edx
	movl	%edx, (%rax)
	movq	-32(%rbp), %rax
	movl	-12(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	$2, -8(%rbp)
	jmp	.L219
.L223:
	nop
.L219:
	jmp	.L221
.L222:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	swapMinHeapNodes, .-swapMinHeapNodes
	.globl	parentOfHeapNode
	.type	parentOfHeapNode, @function
parentOfHeapNode:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L227:
	cmpq	$0, -8(%rbp)
	jne	.L230
	movl	-20(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	jmp	.L229
.L230:
	nop
	jmp	.L227
.L229:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	parentOfHeapNode, .-parentOfHeapNode
	.globl	rightChildOfHeapNode
	.type	rightChildOfHeapNode, @function
rightChildOfHeapNode:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L234:
	cmpq	$0, -8(%rbp)
	jne	.L237
	movl	-20(%rbp), %eax
	addl	$1, %eax
	addl	%eax, %eax
	jmp	.L236
.L237:
	nop
	jmp	.L234
.L236:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	rightChildOfHeapNode, .-rightChildOfHeapNode
	.globl	extractMinElementFromMinHeap
	.type	extractMinElementFromMinHeap, @function
extractMinElementFromMinHeap:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$4, -8(%rbp)
.L253:
	cmpq	$5, -8(%rbp)
	ja	.L254
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L241(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L241(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L241:
	.long	.L246-.L241
	.long	.L245-.L241
	.long	.L244-.L241
	.long	.L255-.L241
	.long	.L242-.L241
	.long	.L255-.L241
	.text
.L242:
	movq	-24(%rbp), %rax
	movl	12(%rax), %eax
	testl	%eax, %eax
	jle	.L247
	movq	$1, -8(%rbp)
	jmp	.L249
.L247:
	movq	$5, -8(%rbp)
	jmp	.L249
.L245:
	movq	-24(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	$1, %eax
	jne	.L250
	movq	$2, -8(%rbp)
	jmp	.L249
.L250:
	movq	$0, -8(%rbp)
	jmp	.L249
.L246:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movq	-24(%rbp), %rax
	movl	12(%rax), %eax
	cltq
	salq	$3, %rax
	subq	$8, %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	(%rdx), %rdx
	movq	%rdx, (%rax)
	movq	-24(%rbp), %rax
	movl	12(%rax), %eax
	leal	-1(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, 12(%rax)
	movq	-24(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	heapify
	movq	$5, -8(%rbp)
	jmp	.L249
.L244:
	movq	-24(%rbp), %rax
	movl	12(%rax), %eax
	leal	-1(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, 12(%rax)
	movq	$3, -8(%rbp)
	jmp	.L249
.L254:
	nop
.L249:
	jmp	.L253
.L255:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	extractMinElementFromMinHeap, .-extractMinElementFromMinHeap
	.globl	initializeMinHeap
	.type	initializeMinHeap, @function
initializeMinHeap:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$0, -16(%rbp)
.L262:
	cmpq	$2, -16(%rbp)
	je	.L263
	cmpq	$2, -16(%rbp)
	ja	.L264
	cmpq	$0, -16(%rbp)
	je	.L259
	cmpq	$1, -16(%rbp)
	jne	.L264
	movl	-28(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-24(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-24(%rbp), %rax
	movl	-28(%rbp), %edx
	movl	%edx, 8(%rax)
	movq	-24(%rbp), %rax
	movl	$0, 12(%rax)
	movq	$2, -16(%rbp)
	jmp	.L260
.L259:
	movq	$1, -16(%rbp)
	jmp	.L260
.L264:
	nop
.L260:
	jmp	.L262
.L263:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	initializeMinHeap, .-initializeMinHeap
	.section	.rodata
.LC19:
	.string	"-> %c "
	.text
	.globl	printPath
	.type	printPath, @function
printPath:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L274:
	cmpq	$3, -8(%rbp)
	je	.L266
	cmpq	$3, -8(%rbp)
	ja	.L275
	cmpq	$0, -8(%rbp)
	je	.L268
	cmpq	$2, -8(%rbp)
	je	.L276
	jmp	.L275
.L266:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	parentArr(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %edi
	call	printPath
	movl	-20(%rbp), %eax
	addl	$65, %eax
	movl	%eax, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L270
.L268:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	parentArr(%rip), %rax
	movl	(%rdx,%rax), %eax
	cmpl	$-1, %eax
	je	.L271
	movq	$3, -8(%rbp)
	jmp	.L270
.L271:
	movq	$2, -8(%rbp)
	jmp	.L270
.L275:
	nop
.L270:
	jmp	.L274
.L276:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	printPath, .-printPath
	.section	.rodata
	.align 8
.LC20:
	.string	"\nWrong choice! Please enter again."
.LC21:
	.string	"\n\nMenu"
	.align 8
.LC22:
	.string	"-----------------------------------------------------"
.LC23:
	.string	"1 - Read File"
.LC24:
	.string	"2 - Show Adjacency Matrix"
.LC25:
	.string	"3 - Find Shortest Path"
.LC26:
	.string	"4 - Exit"
.LC27:
	.string	"\nEnter your choice: "
.LC28:
	.string	"%d"
	.text
	.globl	showMenuAndGetChoice
	.type	showMenuAndGetChoice, @function
showMenuAndGetChoice:
.LFB18:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$10, -16(%rbp)
.L298:
	cmpq	$12, -16(%rbp)
	ja	.L301
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L280(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L280(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L280:
	.long	.L301-.L280
	.long	.L289-.L280
	.long	.L301-.L280
	.long	.L301-.L280
	.long	.L288-.L280
	.long	.L287-.L280
	.long	.L286-.L280
	.long	.L285-.L280
	.long	.L284-.L280
	.long	.L283-.L280
	.long	.L282-.L280
	.long	.L281-.L280
	.long	.L279-.L280
	.text
.L288:
	movl	-24(%rbp), %eax
	cmpl	$4, %eax
	jg	.L290
	movq	$1, -16(%rbp)
	jmp	.L292
.L290:
	movq	$7, -16(%rbp)
	jmp	.L292
.L279:
	movl	-24(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L299
	jmp	.L300
.L284:
	cmpl	$0, -20(%rbp)
	jne	.L294
	movq	$6, -16(%rbp)
	jmp	.L292
.L294:
	movq	$5, -16(%rbp)
	jmp	.L292
.L289:
	movl	$1, -20(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L292
.L281:
	movl	-24(%rbp), %eax
	testl	%eax, %eax
	jle	.L296
	movq	$4, -16(%rbp)
	jmp	.L292
.L296:
	movq	$9, -16(%rbp)
	jmp	.L292
.L283:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L292
.L286:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$11, -16(%rbp)
	jmp	.L292
.L287:
	movl	$10, %edi
	call	putchar@PLT
	movq	$12, -16(%rbp)
	jmp	.L292
.L282:
	movl	$0, -20(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L292
.L285:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L292
.L301:
	nop
.L292:
	jmp	.L298
.L300:
	call	__stack_chk_fail@PLT
.L299:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	showMenuAndGetChoice, .-showMenuAndGetChoice
	.globl	insertElementToMinHeap
	.type	insertElementToMinHeap, @function
insertElementToMinHeap:
.LFB19:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$1, -8(%rbp)
.L320:
	cmpq	$10, -8(%rbp)
	ja	.L321
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L305(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L305(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L305:
	.long	.L321-.L305
	.long	.L311-.L305
	.long	.L321-.L305
	.long	.L322-.L305
	.long	.L321-.L305
	.long	.L309-.L305
	.long	.L321-.L305
	.long	.L308-.L305
	.long	.L307-.L305
	.long	.L306-.L305
	.long	.L304-.L305
	.text
.L307:
	cmpl	$0, -20(%rbp)
	je	.L312
	movq	$7, -8(%rbp)
	jmp	.L314
.L312:
	movq	$3, -8(%rbp)
	jmp	.L314
.L311:
	movq	-40(%rbp), %rax
	movl	12(%rax), %edx
	movq	-40(%rbp), %rax
	movl	8(%rax), %eax
	cmpl	%eax, %edx
	jge	.L315
	movq	$5, -8(%rbp)
	jmp	.L314
.L315:
	movq	$3, -8(%rbp)
	jmp	.L314
.L306:
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	parentOfHeapNode
	movl	%eax, -12(%rbp)
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	(%rax), %rcx
	movl	-20(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rcx, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	swapMinHeapNodes
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	parentOfHeapNode
	movl	%eax, -20(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L314
.L309:
	movq	-40(%rbp), %rax
	movl	12(%rax), %eax
	leal	1(%rax), %edx
	movq	-40(%rbp), %rax
	movl	%edx, 12(%rax)
	movq	-40(%rbp), %rax
	movl	12(%rax), %eax
	subl	$1, %eax
	movl	%eax, -20(%rbp)
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-20(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movq	%rax, (%rdx)
	movq	$8, -8(%rbp)
	jmp	.L314
.L304:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-16(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movl	4(%rax), %edx
	movq	-40(%rbp), %rax
	movq	(%rax), %rcx
	movl	-20(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rcx, %rax
	movl	4(%rax), %eax
	cmpl	%eax, %edx
	jle	.L318
	movq	$9, -8(%rbp)
	jmp	.L314
.L318:
	movq	$3, -8(%rbp)
	jmp	.L314
.L308:
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	parentOfHeapNode
	movl	%eax, -16(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L314
.L321:
	nop
.L314:
	jmp	.L320
.L322:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE19:
	.size	insertElementToMinHeap, .-insertElementToMinHeap
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
