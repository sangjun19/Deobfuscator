	.file	"AkshayaShankariB_DSA_3_flatten.c"
	.text
	.globl	k
	.bss
	.align 4
	.type	k, @object
	.size	k, 4
k:
	.zero	4
	.globl	stack
	.align 16
	.type	stack, @object
	.size	stack, 24
stack:
	.zero	24
	.globl	i
	.align 4
	.type	i, @object
	.size	i, 4
i:
	.zero	4
	.globl	rev
	.align 16
	.type	rev, @object
	.size	rev, 24
rev:
	.zero	24
	.globl	f
	.align 4
	.type	f, @object
	.size	f, 4
f:
	.zero	4
	.globl	choice
	.align 4
	.type	choice, @object
	.size	choice, 4
choice:
	.zero	4
	.globl	_TIG_IZ_rlDj_envp
	.align 8
	.type	_TIG_IZ_rlDj_envp, @object
	.size	_TIG_IZ_rlDj_envp, 8
_TIG_IZ_rlDj_envp:
	.zero	8
	.globl	_TIG_IZ_rlDj_argv
	.align 8
	.type	_TIG_IZ_rlDj_argv, @object
	.size	_TIG_IZ_rlDj_argv, 8
_TIG_IZ_rlDj_argv:
	.zero	8
	.globl	size
	.align 4
	.type	size, @object
	.size	size, 4
size:
	.zero	4
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.globl	num
	.align 4
	.type	num, @object
	.size	num, 4
num:
	.zero	4
	.globl	_TIG_IZ_rlDj_argc
	.align 4
	.type	_TIG_IZ_rlDj_argc, @object
	.size	_TIG_IZ_rlDj_argc, 4
_TIG_IZ_rlDj_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Enter the number to be pushed"
.LC1:
	.string	"%d"
.LC2:
	.string	"Stack overflow"
	.text
	.globl	push
	.type	push, @function
push:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$4, -8(%rbp)
.L11:
	cmpq	$4, -8(%rbp)
	je	.L2
	cmpq	$4, -8(%rbp)
	ja	.L12
	cmpq	$2, -8(%rbp)
	je	.L4
	cmpq	$2, -8(%rbp)
	ja	.L12
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$1, -8(%rbp)
	je	.L13
	jmp	.L12
.L2:
	movl	size(%rip), %eax
	leal	-1(%rax), %edx
	movl	top(%rip), %eax
	cmpl	%eax, %edx
	jne	.L7
	movq	$2, -8(%rbp)
	jmp	.L9
.L7:
	movq	$0, -8(%rbp)
	jmp	.L9
.L5:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	num(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %edx
	movl	num(%rip), %eax
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	leaq	stack(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movq	$1, -8(%rbp)
	jmp	.L9
.L4:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L9
.L12:
	nop
.L9:
	jmp	.L11
.L13:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	push, .-push
	.globl	pali
	.type	pali, @function
pali:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$15, -8(%rbp)
.L36:
	cmpq	$15, -8(%rbp)
	ja	.L38
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L17(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L17(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L17:
	.long	.L27-.L17
	.long	.L38-.L17
	.long	.L38-.L17
	.long	.L26-.L17
	.long	.L25-.L17
	.long	.L24-.L17
	.long	.L23-.L17
	.long	.L22-.L17
	.long	.L38-.L17
	.long	.L21-.L17
	.long	.L38-.L17
	.long	.L38-.L17
	.long	.L20-.L17
	.long	.L19-.L17
	.long	.L18-.L17
	.long	.L16-.L17
	.text
.L25:
	movl	k(%rip), %eax
	subl	$1, %eax
	movl	%eax, k(%rip)
	movq	$12, -8(%rbp)
	jmp	.L28
.L18:
	movl	i(%rip), %eax
	subl	$1, %eax
	movl	%eax, i(%rip)
	movq	$13, -8(%rbp)
	jmp	.L28
.L16:
	movq	$3, -8(%rbp)
	jmp	.L28
.L20:
	movl	i(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %edx
	movl	k(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	rev(%rip), %rax
	movl	(%rcx,%rax), %eax
	cmpl	%eax, %edx
	je	.L29
	movq	$6, -8(%rbp)
	jmp	.L28
.L29:
	movq	$14, -8(%rbp)
	jmp	.L28
.L26:
	movl	$1, -16(%rbp)
	movl	top(%rip), %eax
	movl	%eax, i(%rip)
	movq	$0, -8(%rbp)
	jmp	.L28
.L21:
	movl	-16(%rbp), %eax
	jmp	.L37
.L19:
	movl	i(%rip), %eax
	testl	%eax, %eax
	js	.L32
	movq	$4, -8(%rbp)
	jmp	.L28
.L32:
	movq	$9, -8(%rbp)
	jmp	.L28
.L23:
	movl	$0, -16(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L28
.L24:
	movl	top(%rip), %eax
	movl	%eax, i(%rip)
	movq	$13, -8(%rbp)
	jmp	.L28
.L27:
	movl	i(%rip), %eax
	testl	%eax, %eax
	js	.L34
	movq	$7, -8(%rbp)
	jmp	.L28
.L34:
	movq	$5, -8(%rbp)
	jmp	.L28
.L22:
	movl	k(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	k(%rip), %eax
	addl	$1, %eax
	movl	%eax, k(%rip)
	movl	i(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	-12(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	leaq	rev(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movl	i(%rip), %eax
	subl	$1, %eax
	movl	%eax, i(%rip)
	movq	$0, -8(%rbp)
	jmp	.L28
.L38:
	nop
.L28:
	jmp	.L36
.L37:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	pali, .-pali
	.section	.rodata
.LC3:
	.string	"Enter size for stack"
	.align 8
.LC4:
	.string	"\n-----MENU-----\n1.Push 2.Pop 3.Display 4.Check for Palindrome 5.Exit"
.LC5:
	.string	"It's not a Palindrome"
.LC6:
	.string	"Wrong choice..."
.LC7:
	.string	"Enter the choice"
.LC8:
	.string	"It's Palindrome"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movl	$0, -16(%rbp)
	jmp	.L40
.L41:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	rev(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -16(%rbp)
.L40:
	cmpl	$5, -16(%rbp)
	jle	.L41
	nop
.L42:
	movl	$0, -12(%rbp)
	jmp	.L43
.L44:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -12(%rbp)
.L43:
	cmpl	$5, -12(%rbp)
	jle	.L44
	nop
.L45:
	movl	$0, i(%rip)
	nop
.L46:
	movl	$0, k(%rip)
	nop
.L47:
	movl	$0, num(%rip)
	nop
.L48:
	movl	$-1, top(%rip)
	nop
.L49:
	movl	$0, choice(%rip)
	nop
.L50:
	movl	$0, f(%rip)
	nop
.L51:
	movl	$0, size(%rip)
	nop
.L52:
	movq	$0, _TIG_IZ_rlDj_envp(%rip)
	nop
.L53:
	movq	$0, _TIG_IZ_rlDj_argv(%rip)
	nop
.L54:
	movl	$0, _TIG_IZ_rlDj_argc(%rip)
	nop
	nop
.L55:
.L56:
#APP
# 160 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-rlDj--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_rlDj_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_rlDj_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_rlDj_envp(%rip)
	nop
	movq	$6, -8(%rbp)
.L83:
	cmpq	$17, -8(%rbp)
	ja	.L84
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L59(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L59(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L59:
	.long	.L71-.L59
	.long	.L70-.L59
	.long	.L84-.L59
	.long	.L69-.L59
	.long	.L84-.L59
	.long	.L68-.L59
	.long	.L67-.L59
	.long	.L66-.L59
	.long	.L84-.L59
	.long	.L65-.L59
	.long	.L64-.L59
	.long	.L63-.L59
	.long	.L84-.L59
	.long	.L62-.L59
	.long	.L84-.L59
	.long	.L61-.L59
	.long	.L60-.L59
	.long	.L58-.L59
	.text
.L61:
	call	display
	movq	$5, -8(%rbp)
	jmp	.L72
.L70:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	size(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -8(%rbp)
	jmp	.L72
.L69:
	call	pali
	movl	%eax, f(%rip)
	movq	$17, -8(%rbp)
	jmp	.L72
.L60:
	movl	$0, %edi
	call	exit@PLT
.L63:
	movl	choice(%rip), %eax
	cmpl	$5, %eax
	ja	.L73
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L75(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L75(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L75:
	.long	.L73-.L75
	.long	.L79-.L75
	.long	.L78-.L75
	.long	.L77-.L75
	.long	.L76-.L75
	.long	.L74-.L75
	.text
.L74:
	movq	$16, -8(%rbp)
	jmp	.L80
.L76:
	movq	$3, -8(%rbp)
	jmp	.L80
.L77:
	movq	$15, -8(%rbp)
	jmp	.L80
.L78:
	movq	$7, -8(%rbp)
	jmp	.L80
.L79:
	movq	$10, -8(%rbp)
	jmp	.L80
.L73:
	movq	$13, -8(%rbp)
	nop
.L80:
	jmp	.L72
.L65:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -8(%rbp)
	jmp	.L72
.L62:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -8(%rbp)
	jmp	.L72
.L58:
	movl	f(%rip), %eax
	cmpl	$1, %eax
	jne	.L81
	movq	$0, -8(%rbp)
	jmp	.L72
.L81:
	movq	$9, -8(%rbp)
	jmp	.L72
.L67:
	movq	$1, -8(%rbp)
	jmp	.L72
.L68:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	choice(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$11, -8(%rbp)
	jmp	.L72
.L64:
	call	push
	movq	$5, -8(%rbp)
	jmp	.L72
.L71:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -8(%rbp)
	jmp	.L72
.L66:
	call	pop
	movq	$5, -8(%rbp)
	jmp	.L72
.L84:
	nop
.L72:
	jmp	.L83
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.section	.rodata
.LC9:
	.string	"%d\n"
.LC10:
	.string	"Stack is empty"
.LC11:
	.string	"Stack contents..."
	.text
	.globl	display
	.type	display, @function
display:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L100:
	cmpq	$8, -8(%rbp)
	ja	.L101
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L88(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L88(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L88:
	.long	.L101-.L88
	.long	.L102-.L88
	.long	.L92-.L88
	.long	.L91-.L88
	.long	.L101-.L88
	.long	.L90-.L88
	.long	.L89-.L88
	.long	.L101-.L88
	.long	.L87-.L88
	.text
.L87:
	movl	i(%rip), %eax
	testl	%eax, %eax
	js	.L94
	movq	$6, -8(%rbp)
	jmp	.L96
.L94:
	movq	$1, -8(%rbp)
	jmp	.L96
.L91:
	movl	top(%rip), %eax
	cmpl	$-1, %eax
	jne	.L98
	movq	$5, -8(%rbp)
	jmp	.L96
.L98:
	movq	$2, -8(%rbp)
	jmp	.L96
.L89:
	movl	i(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	i(%rip), %eax
	subl	$1, %eax
	movl	%eax, i(%rip)
	movq	$8, -8(%rbp)
	jmp	.L96
.L90:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L96
.L92:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	top(%rip), %eax
	movl	%eax, i(%rip)
	movq	$8, -8(%rbp)
	jmp	.L96
.L101:
	nop
.L96:
	jmp	.L100
.L102:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	display, .-display
	.section	.rodata
.LC12:
	.string	"Stack underflow"
.LC13:
	.string	"Popped element is %d\n"
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L112:
	cmpq	$3, -8(%rbp)
	je	.L104
	cmpq	$3, -8(%rbp)
	ja	.L113
	cmpq	$2, -8(%rbp)
	je	.L106
	cmpq	$2, -8(%rbp)
	ja	.L113
	cmpq	$0, -8(%rbp)
	je	.L114
	cmpq	$1, -8(%rbp)
	jne	.L113
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L108
.L104:
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, num(%rip)
	movl	num(%rip), %eax
	movl	%eax, %esi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movq	$0, -8(%rbp)
	jmp	.L108
.L106:
	movl	top(%rip), %eax
	cmpl	$-1, %eax
	jne	.L110
	movq	$1, -8(%rbp)
	jmp	.L108
.L110:
	movq	$3, -8(%rbp)
	jmp	.L108
.L113:
	nop
.L108:
	jmp	.L112
.L114:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	pop, .-pop
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
