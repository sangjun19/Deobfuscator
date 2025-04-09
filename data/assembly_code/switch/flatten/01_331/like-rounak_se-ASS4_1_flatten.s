	.file	"like-rounak_se-ASS4_1_flatten.c"
	.text
	.globl	stock_item
	.bss
	.align 32
	.type	stock_item, @object
	.size	stock_item, 76
stock_item:
	.zero	76
	.globl	file_pointer
	.align 8
	.type	file_pointer, @object
	.size	file_pointer, 8
file_pointer:
	.zero	8
	.globl	_TIG_IZ_mjkz_argv
	.align 8
	.type	_TIG_IZ_mjkz_argv, @object
	.size	_TIG_IZ_mjkz_argv, 8
_TIG_IZ_mjkz_argv:
	.zero	8
	.globl	_TIG_IZ_mjkz_argc
	.align 4
	.type	_TIG_IZ_mjkz_argc, @object
	.size	_TIG_IZ_mjkz_argc, 4
_TIG_IZ_mjkz_argc:
	.zero	4
	.globl	_TIG_IZ_mjkz_envp
	.align 8
	.type	_TIG_IZ_mjkz_envp, @object
	.size	_TIG_IZ_mjkz_envp, 8
_TIG_IZ_mjkz_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"1. Buy product"
.LC1:
	.string	"2. View product inventory"
.LC2:
	.string	"Enter your choice: "
.LC3:
	.string	"%d"
.LC4:
	.string	"Invalid input"
	.text
	.globl	customer_interface
	.type	customer_interface, @function
customer_interface:
.LFB1:
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
	movq	$1, -16(%rbp)
.L16:
	cmpq	$8, -16(%rbp)
	ja	.L19
	movq	-16(%rbp), %rax
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
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L20-.L4
	.long	.L7-.L4
	.long	.L19-.L4
	.long	.L19-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L3:
	call	buy_item
	movq	$2, -16(%rbp)
	jmp	.L11
.L9:
	movq	$6, -16(%rbp)
	jmp	.L11
.L7:
	movl	-20(%rbp), %eax
	cmpl	$1, %eax
	je	.L12
	cmpl	$2, %eax
	jne	.L13
	movq	$7, -16(%rbp)
	jmp	.L14
.L12:
	movq	$8, -16(%rbp)
	jmp	.L14
.L13:
	movq	$0, -16(%rbp)
	nop
.L14:
	jmp	.L11
.L6:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$3, -16(%rbp)
	jmp	.L11
.L10:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L11
.L5:
	call	display_items
	movq	$2, -16(%rbp)
	jmp	.L11
.L19:
	nop
.L11:
	jmp	.L16
.L20:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L18
	call	__stack_chk_fail@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	customer_interface, .-customer_interface
	.section	.rodata
	.align 8
.LC5:
	.string	"Select the operation to be performed"
.LC6:
	.string	"1. Update the product name"
.LC7:
	.string	"2. Update the quantity"
.LC8:
	.string	"3. Update the product price"
.LC9:
	.string	"Enter the new product name: "
.LC10:
	.string	"%s"
.LC11:
	.string	"\nProduct not found"
.LC12:
	.string	"<== Update products ==>\n"
	.align 8
.LC13:
	.string	"Enter the product ID to update: "
.LC14:
	.string	"rb+"
.LC15:
	.string	"product.txt"
.LC16:
	.string	"Enter new product quantity: "
	.align 8
.LC17:
	.string	"\nProduct updated successfully..."
.LC18:
	.string	"Enter new product price: "
.LC19:
	.string	"%f"
	.text
	.globl	update_inventory
	.type	update_inventory, @function
update_inventory:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$16, -16(%rbp)
.L53:
	cmpq	$25, -16(%rbp)
	ja	.L56
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L56-.L24
	.long	.L39-.L24
	.long	.L56-.L24
	.long	.L38-.L24
	.long	.L37-.L24
	.long	.L56-.L24
	.long	.L56-.L24
	.long	.L56-.L24
	.long	.L36-.L24
	.long	.L56-.L24
	.long	.L56-.L24
	.long	.L35-.L24
	.long	.L56-.L24
	.long	.L34-.L24
	.long	.L33-.L24
	.long	.L32-.L24
	.long	.L31-.L24
	.long	.L30-.L24
	.long	.L29-.L24
	.long	.L57-.L24
	.long	.L56-.L24
	.long	.L27-.L24
	.long	.L56-.L24
	.long	.L26-.L24
	.long	.L25-.L24
	.long	.L23-.L24
	.text
.L29:
	movl	$1, -28(%rbp)
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$23, -16(%rbp)
	jmp	.L40
.L23:
	cmpq	$1, -24(%rbp)
	jne	.L41
	movq	$4, -16(%rbp)
	jmp	.L40
.L41:
	movq	$1, -16(%rbp)
	jmp	.L40
.L37:
	movl	stock_item(%rip), %edx
	movl	-36(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L43
	movq	$18, -16(%rbp)
	jmp	.L40
.L43:
	movq	$21, -16(%rbp)
	jmp	.L40
.L33:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	4+stock_item(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$3, -16(%rbp)
	jmp	.L40
.L32:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -16(%rbp)
	jmp	.L40
.L36:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -16(%rbp)
	jmp	.L40
.L39:
	cmpl	$1, -28(%rbp)
	jne	.L45
	movq	$13, -16(%rbp)
	jmp	.L40
.L45:
	movq	$15, -16(%rbp)
	jmp	.L40
.L26:
	movl	-32(%rbp), %eax
	cmpl	$3, %eax
	je	.L47
	cmpl	$3, %eax
	jg	.L48
	cmpl	$1, %eax
	je	.L49
	cmpl	$2, %eax
	je	.L50
	jmp	.L48
.L47:
	movq	$17, -16(%rbp)
	jmp	.L51
.L50:
	movq	$11, -16(%rbp)
	jmp	.L51
.L49:
	movq	$14, -16(%rbp)
	jmp	.L51
.L48:
	movq	$8, -16(%rbp)
	nop
.L51:
	jmp	.L40
.L38:
	movq	file_pointer(%rip), %rax
	movl	$1, %edx
	movq	$-76, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	file_pointer(%rip), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$76, %esi
	leaq	stock_item(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	file_pointer(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$1, -16(%rbp)
	jmp	.L40
.L31:
	movq	$24, -16(%rbp)
	jmp	.L40
.L25:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC14(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, file_pointer(%rip)
	movq	$21, -16(%rbp)
	jmp	.L40
.L27:
	movq	file_pointer(%rip), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$76, %esi
	leaq	stock_item(%rip), %rax
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -24(%rbp)
	movq	$25, -16(%rbp)
	jmp	.L40
.L35:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	56+stock_item(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$3, -16(%rbp)
	jmp	.L40
.L34:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -16(%rbp)
	jmp	.L40
.L30:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	60+stock_item(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$3, -16(%rbp)
	jmp	.L40
.L56:
	nop
.L40:
	jmp	.L53
.L57:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L55
	call	__stack_chk_fail@PLT
.L55:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	update_inventory, .-update_inventory
	.section	.rodata
.LC20:
	.string	"%-10d %-30s %-30d %-20f %s\n"
.LC21:
	.string	"cls"
.LC22:
	.string	"<=== Product List ===>\n"
.LC23:
	.string	"Date"
.LC24:
	.string	"Price"
.LC25:
	.string	"Quantity"
.LC26:
	.string	"Product Name"
.LC27:
	.string	"ID"
.LC28:
	.string	"%-10s %-30s %-30s %-20s %s\n"
	.align 8
.LC29:
	.string	"\n---------------------------------------------------------------------------------------"
.LC30:
	.string	"rb"
	.text
	.globl	display_items
	.type	display_items, @function
display_items:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$7, -8(%rbp)
.L72:
	cmpq	$8, -8(%rbp)
	ja	.L73
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L61(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L61(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L61:
	.long	.L67-.L61
	.long	.L73-.L61
	.long	.L66-.L61
	.long	.L65-.L61
	.long	.L73-.L61
	.long	.L74-.L61
	.long	.L63-.L61
	.long	.L62-.L61
	.long	.L60-.L61
	.text
.L60:
	cmpq	$1, -16(%rbp)
	jne	.L68
	movq	$0, -8(%rbp)
	jmp	.L70
.L68:
	movq	$3, -8(%rbp)
	jmp	.L70
.L65:
	movq	file_pointer(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$5, -8(%rbp)
	jmp	.L70
.L63:
	movq	file_pointer(%rip), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$76, %esi
	leaq	stock_item(%rip), %rax
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -16(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L70
.L67:
	movss	60+stock_item(%rip), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rcx
	movl	56+stock_item(%rip), %edx
	movl	stock_item(%rip), %eax
	leaq	64+stock_item(%rip), %r8
	movq	%rcx, %xmm0
	movl	%edx, %ecx
	leaq	4+stock_item(%rip), %rdx
	movl	%eax, %esi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L70
.L62:
	movq	$2, -8(%rbp)
	jmp	.L70
.L66:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	system@PLT
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC23(%rip), %r9
	leaq	.LC24(%rip), %r8
	leaq	.LC25(%rip), %rax
	movq	%rax, %rcx
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC27(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC30(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, file_pointer(%rip)
	movq	$6, -8(%rbp)
	jmp	.L70
.L73:
	nop
.L70:
	jmp	.L72
.L74:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	display_items, .-display_items
	.section	.rodata
.LC31:
	.string	"temp.txt"
.LC32:
	.string	"wb"
	.text
	.globl	delete_item
	.type	delete_item, @function
delete_item:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$3, -8(%rbp)
.L93:
	cmpq	$11, -8(%rbp)
	ja	.L94
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L78(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L78(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L78:
	.long	.L86-.L78
	.long	.L85-.L78
	.long	.L84-.L78
	.long	.L83-.L78
	.long	.L94-.L78
	.long	.L82-.L78
	.long	.L94-.L78
	.long	.L95-.L78
	.long	.L94-.L78
	.long	.L80-.L78
	.long	.L79-.L78
	.long	.L77-.L78
	.text
.L85:
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$76, %esi
	leaq	stock_item(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$2, -8(%rbp)
	jmp	.L87
.L83:
	movq	$5, -8(%rbp)
	jmp	.L87
.L77:
	movq	file_pointer(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	remove@PLT
	leaq	.LC15(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	call	rename@PLT
	movq	$7, -8(%rbp)
	jmp	.L87
.L80:
	movl	$1, -28(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L87
.L82:
	movl	$0, -28(%rbp)
	leaq	.LC30(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, file_pointer(%rip)
	leaq	.LC32(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -24(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L87
.L79:
	movl	stock_item(%rip), %eax
	cmpl	%eax, -36(%rbp)
	jne	.L88
	movq	$9, -8(%rbp)
	jmp	.L87
.L88:
	movq	$1, -8(%rbp)
	jmp	.L87
.L86:
	cmpq	$1, -16(%rbp)
	jne	.L90
	movq	$10, -8(%rbp)
	jmp	.L87
.L90:
	movq	$11, -8(%rbp)
	jmp	.L87
.L84:
	movq	file_pointer(%rip), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$76, %esi
	leaq	stock_item(%rip), %rax
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L87
.L94:
	nop
.L87:
	jmp	.L93
.L95:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	delete_item, .-delete_item
	.section	.rodata
.LC33:
	.string	"1. Add product"
.LC34:
	.string	"2. Update inventory"
.LC35:
	.string	"3. Delete product"
.LC36:
	.string	"4. Display products"
	.text
	.globl	admin_interface
	.type	admin_interface, @function
admin_interface:
.LFB7:
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
	movq	$6, -16(%rbp)
.L116:
	cmpq	$11, -16(%rbp)
	ja	.L119
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L99(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L99(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L99:
	.long	.L120-.L99
	.long	.L119-.L99
	.long	.L106-.L99
	.long	.L119-.L99
	.long	.L105-.L99
	.long	.L104-.L99
	.long	.L103-.L99
	.long	.L102-.L99
	.long	.L101-.L99
	.long	.L119-.L99
	.long	.L100-.L99
	.long	.L98-.L99
	.text
.L105:
	call	update_inventory
	movq	$0, -16(%rbp)
	jmp	.L108
.L101:
	call	delete_item_prompt
	movq	$0, -16(%rbp)
	jmp	.L108
.L98:
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -16(%rbp)
	jmp	.L108
.L103:
	movq	$11, -16(%rbp)
	jmp	.L108
.L104:
	call	add_item
	movq	$0, -16(%rbp)
	jmp	.L108
.L100:
	call	display_items
	movq	$0, -16(%rbp)
	jmp	.L108
.L102:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L108
.L106:
	movl	-20(%rbp), %eax
	cmpl	$4, %eax
	je	.L110
	cmpl	$4, %eax
	jg	.L111
	cmpl	$3, %eax
	je	.L112
	cmpl	$3, %eax
	jg	.L111
	cmpl	$1, %eax
	je	.L113
	cmpl	$2, %eax
	je	.L114
	jmp	.L111
.L110:
	movq	$10, -16(%rbp)
	jmp	.L115
.L112:
	movq	$8, -16(%rbp)
	jmp	.L115
.L114:
	movq	$4, -16(%rbp)
	jmp	.L115
.L113:
	movq	$5, -16(%rbp)
	jmp	.L115
.L111:
	movq	$7, -16(%rbp)
	nop
.L115:
	jmp	.L108
.L119:
	nop
.L108:
	jmp	.L116
.L120:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L118
	call	__stack_chk_fail@PLT
.L118:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	admin_interface, .-admin_interface
	.section	.rodata
.LC37:
	.string	"Product not found"
.LC38:
	.string	"<== Buy products ==>\n"
.LC39:
	.string	"Enter the product ID to buy: "
	.align 8
.LC40:
	.string	"Enter the quantity of the product: "
	.align 8
.LC41:
	.string	"<===== Here is the invoice =====>"
.LC42:
	.string	"Total amount payable: %.2f\n"
	.align 8
.LC43:
	.string	"Product bought successfully..."
	.align 8
.LC44:
	.string	"Insufficient quantity available"
	.text
	.globl	buy_item
	.type	buy_item, @function
buy_item:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$19, -16(%rbp)
.L155:
	cmpq	$22, -16(%rbp)
	ja	.L158
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L124(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L124(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L124:
	.long	.L140-.L124
	.long	.L139-.L124
	.long	.L138-.L124
	.long	.L137-.L124
	.long	.L159-.L124
	.long	.L158-.L124
	.long	.L158-.L124
	.long	.L158-.L124
	.long	.L159-.L124
	.long	.L134-.L124
	.long	.L158-.L124
	.long	.L133-.L124
	.long	.L158-.L124
	.long	.L132-.L124
	.long	.L131-.L124
	.long	.L130-.L124
	.long	.L129-.L124
	.long	.L158-.L124
	.long	.L128-.L124
	.long	.L127-.L124
	.long	.L126-.L124
	.long	.L125-.L124
	.long	.L123-.L124
	.text
.L128:
	cmpl	$1, -32(%rbp)
	jne	.L141
	movq	$21, -16(%rbp)
	jmp	.L143
.L141:
	movq	$14, -16(%rbp)
	jmp	.L143
.L131:
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L143
.L130:
	movl	$1, -32(%rbp)
	movss	60+stock_item(%rip), %xmm0
	movss	%xmm0, -28(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L143
.L139:
	movl	$0, -32(%rbp)
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC40(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC14(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, file_pointer(%rip)
	movq	$3, -16(%rbp)
	jmp	.L143
.L137:
	movq	file_pointer(%rip), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$76, %esi
	leaq	stock_item(%rip), %rax
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L143
.L129:
	movl	stock_item(%rip), %eax
	movl	%eax, %edi
	call	delete_item
	movq	$18, -16(%rbp)
	jmp	.L143
.L125:
	leaq	.LC41(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-36(%rbp), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	mulss	-28(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC42(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L143
.L133:
	movl	56+stock_item(%rip), %eax
	movl	-36(%rbp), %edx
	subl	%edx, %eax
	testl	%eax, %eax
	js	.L145
	movq	$2, -16(%rbp)
	jmp	.L143
.L145:
	movq	$3, -16(%rbp)
	jmp	.L143
.L134:
	cmpq	$1, -24(%rbp)
	jne	.L147
	movq	$22, -16(%rbp)
	jmp	.L143
.L147:
	movq	$18, -16(%rbp)
	jmp	.L143
.L132:
	leaq	.LC44(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -16(%rbp)
	jmp	.L143
.L127:
	movq	$1, -16(%rbp)
	jmp	.L143
.L123:
	movl	stock_item(%rip), %edx
	movl	-40(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L149
	movq	$15, -16(%rbp)
	jmp	.L143
.L149:
	movq	$3, -16(%rbp)
	jmp	.L143
.L140:
	movl	56+stock_item(%rip), %eax
	movl	-36(%rbp), %edx
	subl	%edx, %eax
	testl	%eax, %eax
	jns	.L151
	movq	$13, -16(%rbp)
	jmp	.L143
.L151:
	movq	$11, -16(%rbp)
	jmp	.L143
.L138:
	movl	56+stock_item(%rip), %eax
	movl	-36(%rbp), %edx
	subl	%edx, %eax
	movl	%eax, 56+stock_item(%rip)
	movq	file_pointer(%rip), %rax
	movl	$1, %edx
	movq	$-76, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	file_pointer(%rip), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$76, %esi
	leaq	stock_item(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	file_pointer(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$20, -16(%rbp)
	jmp	.L143
.L126:
	movl	56+stock_item(%rip), %eax
	testl	%eax, %eax
	jne	.L153
	movq	$16, -16(%rbp)
	jmp	.L143
.L153:
	movq	$18, -16(%rbp)
	jmp	.L143
.L158:
	nop
.L143:
	jmp	.L155
.L159:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L157
	call	__stack_chk_fail@PLT
.L157:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	buy_item, .-buy_item
	.section	.rodata
	.align 8
.LC45:
	.string	"Product deleted successfully..."
.LC46:
	.string	"<== Delete Products ==>\n"
	.align 8
.LC47:
	.string	"Enter the product ID to delete: "
	.text
	.globl	delete_item_prompt
	.type	delete_item_prompt, @function
delete_item_prompt:
.LFB9:
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
	movq	$8, -16(%rbp)
.L181:
	cmpq	$13, -16(%rbp)
	ja	.L184
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L163(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L163(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L163:
	.long	.L184-.L163
	.long	.L172-.L163
	.long	.L171-.L163
	.long	.L184-.L163
	.long	.L184-.L163
	.long	.L184-.L163
	.long	.L170-.L163
	.long	.L169-.L163
	.long	.L168-.L163
	.long	.L167-.L163
	.long	.L166-.L163
	.long	.L165-.L163
	.long	.L185-.L163
	.long	.L162-.L163
	.text
.L168:
	movq	$2, -16(%rbp)
	jmp	.L174
.L172:
	movl	$1, -28(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L174
.L165:
	cmpl	$1, -28(%rbp)
	jne	.L175
	movq	$7, -16(%rbp)
	jmp	.L174
.L175:
	movq	$13, -16(%rbp)
	jmp	.L174
.L167:
	cmpq	$1, -24(%rbp)
	jne	.L177
	movq	$10, -16(%rbp)
	jmp	.L174
.L177:
	movq	$11, -16(%rbp)
	jmp	.L174
.L162:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -16(%rbp)
	jmp	.L174
.L170:
	movq	file_pointer(%rip), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$76, %esi
	leaq	stock_item(%rip), %rax
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L174
.L166:
	movl	stock_item(%rip), %edx
	movl	-32(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L179
	movq	$1, -16(%rbp)
	jmp	.L174
.L179:
	movq	$6, -16(%rbp)
	jmp	.L174
.L169:
	leaq	.LC45(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-32(%rbp), %eax
	movl	%eax, %edi
	call	delete_item
	movq	$12, -16(%rbp)
	jmp	.L174
.L171:
	movl	$0, -28(%rbp)
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC47(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC30(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, file_pointer(%rip)
	movq	$6, -16(%rbp)
	jmp	.L174
.L184:
	nop
.L174:
	jmp	.L181
.L185:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L183
	call	__stack_chk_fail@PLT
.L183:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	delete_item_prompt, .-delete_item_prompt
	.section	.rodata
.LC49:
	.string	"1. Administrator"
.LC50:
	.string	"2. Customer"
.LC51:
	.string	"0. Exit"
.LC52:
	.string	"--> Enter your choice: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB12:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, file_pointer(%rip)
	nop
.L187:
	movl	$0, stock_item(%rip)
	movb	$0, 4+stock_item(%rip)
	movb	$0, 5+stock_item(%rip)
	movb	$0, 6+stock_item(%rip)
	movb	$0, 7+stock_item(%rip)
	movb	$0, 8+stock_item(%rip)
	movb	$0, 9+stock_item(%rip)
	movb	$0, 10+stock_item(%rip)
	movb	$0, 11+stock_item(%rip)
	movb	$0, 12+stock_item(%rip)
	movb	$0, 13+stock_item(%rip)
	movb	$0, 14+stock_item(%rip)
	movb	$0, 15+stock_item(%rip)
	movb	$0, 16+stock_item(%rip)
	movb	$0, 17+stock_item(%rip)
	movb	$0, 18+stock_item(%rip)
	movb	$0, 19+stock_item(%rip)
	movb	$0, 20+stock_item(%rip)
	movb	$0, 21+stock_item(%rip)
	movb	$0, 22+stock_item(%rip)
	movb	$0, 23+stock_item(%rip)
	movb	$0, 24+stock_item(%rip)
	movb	$0, 25+stock_item(%rip)
	movb	$0, 26+stock_item(%rip)
	movb	$0, 27+stock_item(%rip)
	movb	$0, 28+stock_item(%rip)
	movb	$0, 29+stock_item(%rip)
	movb	$0, 30+stock_item(%rip)
	movb	$0, 31+stock_item(%rip)
	movb	$0, 32+stock_item(%rip)
	movb	$0, 33+stock_item(%rip)
	movb	$0, 34+stock_item(%rip)
	movb	$0, 35+stock_item(%rip)
	movb	$0, 36+stock_item(%rip)
	movb	$0, 37+stock_item(%rip)
	movb	$0, 38+stock_item(%rip)
	movb	$0, 39+stock_item(%rip)
	movb	$0, 40+stock_item(%rip)
	movb	$0, 41+stock_item(%rip)
	movb	$0, 42+stock_item(%rip)
	movb	$0, 43+stock_item(%rip)
	movb	$0, 44+stock_item(%rip)
	movb	$0, 45+stock_item(%rip)
	movb	$0, 46+stock_item(%rip)
	movb	$0, 47+stock_item(%rip)
	movb	$0, 48+stock_item(%rip)
	movb	$0, 49+stock_item(%rip)
	movb	$0, 50+stock_item(%rip)
	movb	$0, 51+stock_item(%rip)
	movb	$0, 52+stock_item(%rip)
	movb	$0, 53+stock_item(%rip)
	movl	$0, 56+stock_item(%rip)
	pxor	%xmm0, %xmm0
	movss	%xmm0, 60+stock_item(%rip)
	movb	$0, 64+stock_item(%rip)
	movb	$0, 65+stock_item(%rip)
	movb	$0, 66+stock_item(%rip)
	movb	$0, 67+stock_item(%rip)
	movb	$0, 68+stock_item(%rip)
	movb	$0, 69+stock_item(%rip)
	movb	$0, 70+stock_item(%rip)
	movb	$0, 71+stock_item(%rip)
	movb	$0, 72+stock_item(%rip)
	movb	$0, 73+stock_item(%rip)
	movb	$0, 74+stock_item(%rip)
	movb	$0, 75+stock_item(%rip)
	nop
.L188:
	movq	$0, _TIG_IZ_mjkz_envp(%rip)
	nop
.L189:
	movq	$0, _TIG_IZ_mjkz_argv(%rip)
	nop
.L190:
	movl	$0, _TIG_IZ_mjkz_argc(%rip)
	nop
	nop
.L191:
.L192:
#APP
# 178 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-mjkz--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_mjkz_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_mjkz_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_mjkz_envp(%rip)
	nop
	movq	$8, -16(%rbp)
.L208:
	cmpq	$9, -16(%rbp)
	ja	.L210
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L195(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L195(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L195:
	.long	.L210-.L195
	.long	.L210-.L195
	.long	.L201-.L195
	.long	.L200-.L195
	.long	.L199-.L195
	.long	.L198-.L195
	.long	.L197-.L195
	.long	.L210-.L195
	.long	.L196-.L195
	.long	.L194-.L195
	.text
.L199:
	call	admin_interface
	movq	$9, -16(%rbp)
	jmp	.L202
.L196:
	movq	$9, -16(%rbp)
	jmp	.L202
.L200:
	movl	-20(%rbp), %eax
	cmpl	$2, %eax
	je	.L203
	cmpl	$2, %eax
	jg	.L204
	testl	%eax, %eax
	je	.L205
	cmpl	$1, %eax
	je	.L206
	jmp	.L204
.L205:
	movq	$2, -16(%rbp)
	jmp	.L207
.L203:
	movq	$6, -16(%rbp)
	jmp	.L207
.L206:
	movq	$4, -16(%rbp)
	jmp	.L207
.L204:
	movq	$5, -16(%rbp)
	nop
.L207:
	jmp	.L202
.L194:
	leaq	.LC49(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC51(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC52(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$3, -16(%rbp)
	jmp	.L202
.L197:
	call	customer_interface
	movq	$9, -16(%rbp)
	jmp	.L202
.L198:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -16(%rbp)
	jmp	.L202
.L201:
	movl	$0, %edi
	call	exit@PLT
.L210:
	nop
.L202:
	jmp	.L208
	.cfi_endproc
.LFE12:
	.size	main, .-main
	.section	.rodata
.LC53:
	.string	"%02d/%02d/%d"
.LC54:
	.string	"ab"
.LC55:
	.string	"Enter product ID: "
.LC56:
	.string	"Enter the product name: "
.LC57:
	.string	"Enter product quantity: "
.LC58:
	.string	"Enter the product price: "
	.align 8
.LC59:
	.string	"\nProduct added successfully..."
	.text
	.globl	add_item
	.type	add_item, @function
add_item:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$120, %rsp
	.cfi_offset 3, -24
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$0, -120(%rbp)
.L217:
	cmpq	$2, -120(%rbp)
	je	.L220
	cmpq	$2, -120(%rbp)
	ja	.L221
	cmpq	$0, -120(%rbp)
	je	.L214
	cmpq	$1, -120(%rbp)
	jne	.L221
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -112(%rbp)
	movq	-112(%rbp), %rax
	movq	%rax, -128(%rbp)
	leaq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	localtime@PLT
	movq	%rax, -104(%rbp)
	movq	-104(%rbp), %rax
	movq	(%rax), %rcx
	movq	8(%rax), %rbx
	movq	%rcx, -96(%rbp)
	movq	%rbx, -88(%rbp)
	movq	16(%rax), %rcx
	movq	24(%rax), %rbx
	movq	%rcx, -80(%rbp)
	movq	%rbx, -72(%rbp)
	movq	32(%rax), %rcx
	movq	40(%rax), %rbx
	movq	%rcx, -64(%rbp)
	movq	%rbx, -56(%rbp)
	movq	48(%rax), %rax
	movq	%rax, -48(%rbp)
	movl	-76(%rbp), %eax
	leal	1900(%rax), %ecx
	movl	-84(%rbp), %edx
	movl	-80(%rbp), %eax
	leal	1(%rax), %esi
	leaq	-36(%rbp), %rax
	movl	%ecx, %r8d
	movl	%edx, %ecx
	movl	%esi, %edx
	leaq	.LC53(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	64+stock_item(%rip), %rax
	movq	%rax, %rdi
	call	strcpy@PLT
	leaq	.LC54(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, file_pointer(%rip)
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	stock_item(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC56(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	4+stock_item(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC57(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	56+stock_item(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC58(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	60+stock_item(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC59(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	file_pointer(%rip), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$76, %esi
	leaq	stock_item(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	file_pointer(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$2, -120(%rbp)
	jmp	.L215
.L214:
	movq	$1, -120(%rbp)
	jmp	.L215
.L221:
	nop
.L215:
	jmp	.L217
.L220:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L219
	call	__stack_chk_fail@PLT
.L219:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	add_item, .-add_item
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
